from __future__ import annotations

import math
from dataclasses import replace

from .models import (
    ContinuityCandidateAudit,
    ContinuityChainAudit,
    DialogueChain,
    DialogueChainNode,
    GraphEdgeSnapshot,
    GraphSnapshot,
    LocalContinuityScore,
    MemberCommunityConfig,
    MessageSnapshot,
    RelatednessConfig,
)


def _smoothstep(value: float) -> float:
    bounded = min(1.0, max(0.0, value))
    return bounded * bounded * (3.0 - 2.0 * bounded)


def continuity_time_penalty(
    seconds: float,
    *,
    soft_seconds: float = 300.0,
    window_seconds: float = 900.0,
    soft_floor: float = 0.8,
) -> float:
    """Return a monotonic dialogue penalty with a hard upper window."""
    elapsed = max(0.0, seconds)
    if elapsed >= window_seconds:
        return 0.0
    if elapsed <= soft_seconds:
        return 1.0 - (1.0 - soft_floor) * _smoothstep(elapsed / soft_seconds)
    progress = (elapsed - soft_seconds) / (window_seconds - soft_seconds)
    return soft_floor * (1.0 - _smoothstep(progress))


def _cosine_similarity(
    source: tuple[tuple[str, float], ...],
    target: tuple[tuple[str, float], ...],
) -> float:
    if not source or not target:
        return 0.0
    source_weights = dict(source)
    target_weights = dict(target)
    if len(source_weights) > len(target_weights):
        source_weights, target_weights = target_weights, source_weights
    dot_product = sum(
        value * target_weights.get(term, 0.0)
        for term, value in source_weights.items()
    )
    if dot_product <= 0.0:
        return 0.0
    source_norm = math.sqrt(sum(value * value for value in source_weights.values()))
    target_norm = math.sqrt(sum(value * value for value in target_weights.values()))
    denominator = source_norm * target_norm
    return min(1.0, dot_product / denominator) if denominator > 0.0 else 0.0


def _edge_key(source_id: str, target_id: str) -> tuple[str, str]:
    return (
        (source_id, target_id)
        if source_id <= target_id
        else (target_id, source_id)
    )


class DialogueChainModel:
    """Mutable per-member dialogue chains backed by read-only group evidence."""

    __slots__ = ("config", "relation_config", "_chains")

    def __init__(
        self,
        config: MemberCommunityConfig,
        relation_config: RelatednessConfig,
        chains: tuple[DialogueChain, ...] = (),
    ) -> None:
        if sum(chain.active for chain in chains) > 1:
            raise ValueError("only one dialogue chain may be active")
        self.config = config
        self.relation_config = relation_config
        self._chains = list(chains)

    @property
    def chains(self) -> tuple[DialogueChain, ...]:
        return tuple(self._chains)

    @property
    def active_chain(self) -> DialogueChain | None:
        return next((chain for chain in reversed(self._chains) if chain.active), None)

    @property
    def message_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(
            node.message_id
            for chain in self._chains
            for node in chain.nodes
        ))

    def replace_chains(self, chains: tuple[DialogueChain, ...]) -> None:
        if sum(chain.active for chain in chains) > 1:
            raise ValueError("only one dialogue chain may be active")
        self._chains = list(chains)

    def reconcile_ids(
        self,
        available_ids: frozenset[str],
        timestamp: float,
    ) -> tuple[str, ...]:
        kept: list[DialogueChain] = []
        removed: list[str] = []
        for chain in self._chains:
            nodes = tuple(
                node for node in chain.nodes if node.message_id in available_ids
            )
            if not nodes or (
                timestamp - max(node.timestamp for node in nodes)
                >= self.config.continuity_time_window_seconds
            ):
                removed.append(chain.chain_id)
                continue
            kept.append(replace(chain, nodes=nodes))
        self._chains = kept
        return tuple(removed)

    def process(
        self,
        snapshot: GraphSnapshot,
        *,
        source_id: str,
        member_id: str,
    ) -> LocalContinuityScore:
        nodes = {node.msg_id: node for node in snapshot.nodes}
        source = nodes.get(source_id)
        if source is None:
            return LocalContinuityScore(source_id=source_id)
        edges = {
            _edge_key(edge.source_id, edge.target_id): edge
            for edge in snapshot.edges
        }
        removed = self._prune(nodes, source.timestamp)

        if source.user_id == member_id:
            result = self._record_seed(source, removed)
        else:
            reply_target = nodes.get(source.reply_to or "")
            if reply_target is not None and reply_target.user_id == member_id:
                result = self._create_reply_chain(
                    snapshot,
                    nodes,
                    edges,
                    source,
                    reply_target,
                    removed,
                )
            elif member_id in source.mention_ids:
                result = self._record_seed(source, removed)
            else:
                result = self._process_normal(nodes, edges, source, removed)

        if source.text.strip():
            return result
        return replace(result, score=0.0)

    def record_member_output(
        self,
        snapshot: GraphSnapshot,
        *,
        source_id: str,
        member_id: str,
    ) -> LocalContinuityScore:
        nodes = {node.msg_id: node for node in snapshot.nodes}
        source = nodes.get(source_id)
        if source is None or source.user_id != member_id:
            raise ValueError("member message does not exist or belongs to another user")
        removed = self._prune(nodes, source.timestamp)
        return self._record_seed(source, removed)

    def reconcile(self, snapshot: GraphSnapshot, timestamp: float) -> tuple[str, ...]:
        nodes = {node.msg_id: node for node in snapshot.nodes}
        return self._prune(nodes, timestamp)

    def _prune(
        self,
        nodes: dict[str, MessageSnapshot],
        timestamp: float,
    ) -> tuple[str, ...]:
        kept: list[DialogueChain] = []
        removed: list[str] = []
        for chain in self._chains:
            chain_nodes = tuple(
                node for node in chain.nodes if node.message_id in nodes
            )
            if not chain_nodes:
                removed.append(chain.chain_id)
                continue
            latest = max(node.timestamp for node in chain_nodes)
            if timestamp - latest >= self.config.continuity_time_window_seconds:
                removed.append(chain.chain_id)
                continue
            kept.append(replace(chain, nodes=chain_nodes))
        self._chains = kept
        return tuple(removed)

    def _record_seed(
        self,
        source: MessageSnapshot,
        removed: tuple[str, ...],
    ) -> LocalContinuityScore:
        active_index = next(
            (index for index, chain in reversed(tuple(enumerate(self._chains))) if chain.active),
            None,
        )
        node = DialogueChainNode(source.msg_id, 1.0, source.timestamp)
        if active_index is None:
            chain = DialogueChain(
                chain_id=source.msg_id,
                active=True,
                nodes=(node,),
            )
            self._chains.append(chain)
            active_index = len(self._chains) - 1
        else:
            chain = self._chains[active_index]
            if all(item.message_id != source.msg_id for item in chain.nodes):
                chain = replace(chain, nodes=chain.nodes + (node,), low_streak=0)
                self._chains[active_index] = chain
        chain = self._chains[active_index]
        if len(chain.nodes) >= self.config.continuity_chain_capacity:
            chain = replace(
                chain,
                active=False,
                progress=self.config.continuity_chain_capacity,
            )
            self._chains[active_index] = chain
        audit = ContinuityChainAudit(
            chain_id=chain.chain_id,
            active=chain.active,
            contribution=1.0,
        )
        return LocalContinuityScore(
            source_id=source.msg_id,
            score=1.0,
            seed_id=source.msg_id,
            path_ids=(source.msg_id,),
            direct_strength=1.0,
            path_strength=1.0,
            temporal=1.0,
            winning_chain_id=chain.chain_id,
            chains=(audit,),
            admitted=True,
            removed_chain_ids=removed,
        )

    def _create_reply_chain(
        self,
        snapshot: GraphSnapshot,
        nodes: dict[str, MessageSnapshot],
        edges: dict[tuple[str, str], GraphEdgeSnapshot],
        source: MessageSnapshot,
        target: MessageSnapshot,
        removed: tuple[str, ...],
    ) -> LocalContinuityScore:
        self._chains = [
            replace(chain, active=False) if chain.active else chain
            for chain in self._chains
        ]
        chain = DialogueChain(
            chain_id=source.msg_id,
            active=True,
            nodes=(DialogueChainNode(target.msg_id, 1.0, target.timestamp),),
        )
        interval = sorted(
            (
                node for node in snapshot.nodes
                if target.sequence < node.sequence < source.sequence
            ),
            key=lambda node: node.sequence,
        )
        for intermediate in interval:
            contribution, _ = self._chain_contribution(
                chain,
                intermediate,
                nodes,
                edges,
            )
            if contribution >= self.config.continuity_admission_threshold:
                chain = replace(
                    chain,
                    nodes=chain.nodes + (
                        DialogueChainNode(
                            intermediate.msg_id,
                            contribution,
                            intermediate.timestamp,
                        ),
                    ),
                )
        chain = replace(
            chain,
            nodes=chain.nodes + (
                DialogueChainNode(source.msg_id, 1.0, source.timestamp),
            ),
        )
        if len(chain.nodes) >= self.config.continuity_chain_capacity:
            chain = replace(
                chain,
                active=False,
                progress=self.config.continuity_chain_capacity,
            )
        self._chains.append(chain)
        audit = ContinuityChainAudit(
            chain_id=chain.chain_id,
            active=chain.active,
            contribution=1.0,
        )
        return LocalContinuityScore(
            source_id=source.msg_id,
            score=1.0,
            seed_id=target.msg_id,
            path_ids=(source.msg_id, target.msg_id),
            direct_strength=1.0,
            path_strength=1.0,
            temporal=1.0,
            winning_chain_id=chain.chain_id,
            chains=(audit,),
            admitted=True,
            removed_chain_ids=removed,
        )

    def _process_normal(
        self,
        nodes: dict[str, MessageSnapshot],
        edges: dict[tuple[str, str], GraphEdgeSnapshot],
        source: MessageSnapshot,
        removed_before: tuple[str, ...],
    ) -> LocalContinuityScore:
        audits: list[ContinuityChainAudit] = []
        updated: list[DialogueChain] = []
        removed = list(removed_before)
        admitted = False
        for chain in self._chains:
            contribution, candidates = self._chain_contribution(
                chain,
                source,
                nodes,
                edges,
            )
            audits.append(ContinuityChainAudit(
                chain_id=chain.chain_id,
                active=chain.active,
                contribution=contribution,
                candidates=candidates,
            ))
            low_streak = (
                0
                if contribution >= self.config.continuity_admission_threshold
                else chain.low_streak + 1
            )
            progress = chain.progress + (0 if chain.active else 1)
            chain_nodes = chain.nodes
            active = chain.active
            if active and contribution >= self.config.continuity_admission_threshold:
                if all(node.message_id != source.msg_id for node in chain_nodes):
                    chain_nodes = chain_nodes + (
                        DialogueChainNode(
                            source.msg_id,
                            contribution,
                            source.timestamp,
                        ),
                    )
                    admitted = True
                if len(chain_nodes) >= self.config.continuity_chain_capacity:
                    active = False
                    progress = self.config.continuity_chain_capacity
            refreshed = replace(
                chain,
                active=active,
                nodes=chain_nodes,
                progress=progress,
                low_streak=low_streak,
            )
            expired = (
                refreshed.low_streak >= self.config.continuity_low_streak_limit
                or (
                    not refreshed.active
                    and refreshed.progress
                    >= self.config.continuity_static_progress_limit
                )
            )
            if expired:
                removed.append(refreshed.chain_id)
            else:
                updated.append(refreshed)
        self._chains = updated

        winning = max(
            audits,
            key=lambda item: (item.contribution, item.chain_id),
            default=None,
        )
        contribution = winning.contribution if winning is not None else 0.0
        strongest = (
            max(
                winning.candidates,
                key=lambda item: (item.candidate, item.message_id),
                default=None,
            )
            if winning is not None
            else None
        )
        return LocalContinuityScore(
            source_id=source.msg_id,
            score=contribution,
            seed_id=strongest.message_id if strongest is not None else None,
            path_ids=(
                (source.msg_id, strongest.message_id)
                if strongest is not None
                else ()
            ),
            edge_strengths=(
                tuple(item.base_similarity for item in winning.candidates)
                if winning is not None
                else ()
            ),
            direct_strength=contribution,
            path_strength=contribution,
            elapsed_seconds=(
                max(0.0, source.timestamp - nodes[strongest.message_id].timestamp)
                if strongest is not None
                else 0.0
            ),
            temporal=strongest.time_penalty if strongest is not None else 0.0,
            winning_chain_id=winning.chain_id if winning is not None else None,
            chains=tuple(audits),
            admitted=admitted,
            removed_chain_ids=tuple(dict.fromkeys(removed)),
        )

    def _chain_contribution(
        self,
        chain: DialogueChain,
        source: MessageSnapshot,
        nodes: dict[str, MessageSnapshot],
        edges: dict[tuple[str, str], GraphEdgeSnapshot],
    ) -> tuple[float, tuple[ContinuityCandidateAudit, ...]]:
        candidates: list[ContinuityCandidateAudit] = []
        for parent in chain.nodes:
            parent_node = nodes.get(parent.message_id)
            if parent_node is None or parent.message_id == source.msg_id:
                continue
            elapsed = source.timestamp - parent.timestamp
            if elapsed < 0.0:
                continue
            edge = edges.get(_edge_key(source.msg_id, parent.message_id))
            if edge is not None:
                base_similarity = edge.weight
                relation_source = "group_edge"
            else:
                base_similarity = (
                    self.relation_config.text_weight
                    * _cosine_similarity(
                        source.vector_weights,
                        parent_node.vector_weights,
                    )
                )
                relation_source = "cached_cosine"
            if base_similarity < self.config.continuity_min_link_strength:
                continue
            time_penalty = continuity_time_penalty(
                elapsed,
                soft_seconds=self.config.continuity_time_soft_seconds,
                window_seconds=self.config.continuity_time_window_seconds,
                soft_floor=self.config.continuity_time_soft_floor,
            )
            candidate = (
                base_similarity
                * parent.weight
                * self.config.continuity_propagation_decay
                * time_penalty
            )
            if candidate <= 0.0:
                continue
            candidates.append(ContinuityCandidateAudit(
                message_id=parent.message_id,
                relation_source=relation_source,
                base_similarity=base_similarity,
                parent_chain_weight=parent.weight,
                time_penalty=time_penalty,
                candidate=candidate,
            ))
        denominator = sum(item.candidate for item in candidates)
        if denominator <= 0.0:
            return 0.0, tuple(candidates)
        raw = sum(item.candidate * item.candidate for item in candidates) / denominator
        return min(1.0, max(0.0, raw)), tuple(candidates)
