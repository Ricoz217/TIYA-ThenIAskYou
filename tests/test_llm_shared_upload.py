import asyncio
from io import BytesIO

import httpx
import pytest
from PIL import Image

from TIYA.llm.media import MediaService
from TIYA.llm.model import MediaRef
from TIYA.llm.state import StateStore


@pytest.fixture
def image_bytes(monkeypatch):
    output = BytesIO()
    Image.new("RGB", (2, 2)).save(output, "PNG")
    content = output.getvalue()

    async def read(self, ref, gateway=None):
        return content

    monkeypatch.setattr(MediaService, "content", read)
    return content


def test_services_share_the_same_upload_result_without_polling(tmp_path, image_bytes, monkeypatch):
    async def run():
        first = MediaService(StateStore(tmp_path / "state.db"))
        second = MediaService(StateStore(tmp_path / "state.db"))
        started = asyncio.Event()
        release = asyncio.Event()
        calls = []

        async def send(wire):
            calls.append(wire)
            started.set()
            await release.wait()
            return {"id": "shared"}

        async def forbidden(*args):
            pytest.fail("an in-flight local task must not poll another SQLite instance")

        settings = {"endpoint": "https://files"}
        ref = MediaRef(hash_name="local")
        owner = asyncio.create_task(first.upload(ref, settings, send))
        await started.wait()
        monkeypatch.setattr(second, "_meta", forbidden)
        waiter = asyncio.create_task(second.upload(ref, settings, send))
        await asyncio.sleep(0)
        release.set()
        a, b = await asyncio.gather(owner, waiter)
        assert a is b
        assert len(calls) == 1

    asyncio.run(run())


def test_shared_failure_is_not_reuploaded_by_waiters_and_registry_is_cleaned(tmp_path, image_bytes):
    async def run():
        service = MediaService(StateStore(tmp_path / "state.db"))
        started = asyncio.Event()
        release = asyncio.Event()
        error = RuntimeError("upload failed")
        count = 0

        async def send(wire):
            nonlocal count
            count += 1
            started.set()
            await release.wait()
            raise error

        settings = {"endpoint": "https://files"}
        ref = MediaRef(hash_name="local")
        owner = asyncio.create_task(service.upload(ref, settings, send))
        await started.wait()
        waiter = asyncio.create_task(service.upload(ref, settings, send))
        await asyncio.sleep(0)
        release.set()
        results = await asyncio.gather(owner, waiter, return_exceptions=True)
        assert results[0] is results[1] is error
        assert count == 1
        with pytest.raises(RuntimeError):
            await service.upload(ref, settings, send)

        assert count == 2

    asyncio.run(run())


@pytest.mark.parametrize("exit_kind", ["cancel", "timeout"])
def test_first_waiter_exit_does_not_stop_remaining_waiter(tmp_path, image_bytes, exit_kind):
    async def run():
        service = MediaService(StateStore(tmp_path / "state.db"))
        started = asyncio.Event()
        release = asyncio.Event()
        count = 0

        async def send(wire):
            nonlocal count
            count += 1
            started.set()
            await release.wait()
            return {"id": "shared"}

        settings = {"endpoint": "https://files"}
        ref = MediaRef(hash_name="local")
        timeout = 0.2 if exit_kind == "timeout" else 5
        owner = asyncio.create_task(service.upload(ref, settings, send, timeout=timeout))
        await started.wait()
        waiter = asyncio.create_task(service.upload(ref, settings, send, timeout=5))
        await asyncio.sleep(0)
        if exit_kind == "cancel":
            owner.cancel()

        with pytest.raises(asyncio.CancelledError if exit_kind == "cancel" else TimeoutError):
            await owner

        assert not waiter.done()
        release.set()
        assert (await waiter).value == "shared"
        assert count == 1

    asyncio.run(run())


def test_last_waiter_cancel_releases_lease_and_a_new_call_can_upload(tmp_path, image_bytes):
    async def run():
        state = StateStore(tmp_path / "state.db")
        service = MediaService(state)
        started = asyncio.Event()
        stopped = asyncio.Event()

        async def send(wire):
            started.set()
            try:
                await asyncio.Event().wait()

            finally:
                stopped.set()

        settings = {"endpoint": "https://files"}
        ref = MediaRef(hash_name="local")
        owner = asyncio.create_task(service.upload(ref, settings, send))
        await started.wait()
        owner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await owner

        assert stopped.is_set()
        with state.db.transaction() as connection:
            assert connection.execute("SELECT count(*) FROM leases").fetchone()[0] == 0

        async def successful(wire):
            return {"id": "new"}

        assert (await service.upload(ref, settings, successful)).value == "new"

    asyncio.run(run())


def test_force_joins_an_upload_already_in_progress(tmp_path, image_bytes):
    async def run():
        service = MediaService(StateStore(tmp_path / "state.db"))
        started = asyncio.Event()
        release = asyncio.Event()
        count = 0

        async def send(wire):
            nonlocal count
            count += 1
            started.set()
            await release.wait()
            return {"id": f"file-{count}"}

        settings = {"endpoint": "https://files"}
        ref = MediaRef(hash_name="local")
        owner = asyncio.create_task(service.upload(ref, settings, send))
        await started.wait()
        waiter = asyncio.create_task(service.upload(ref, settings, send, force=True))
        await asyncio.sleep(0)
        release.set()
        a, b = await asyncio.gather(owner, waiter)
        assert a is b
        assert count == 1
        assert (await service.upload(ref, settings, send, force=True)).value == "file-2"

    asyncio.run(run())


def test_client_close_is_deferred_until_shared_upload_finishes(tmp_path, image_bytes):
    from TIYA.llm.media import UploadSender
    from TIYA.llm.gateway import Gateway

    async def run():
        service = MediaService(StateStore(tmp_path / "state.db"))
        started = asyncio.Event()
        release = asyncio.Event()

        async def handler(request):
            started.set()
            await release.wait()
            return httpx.Response(200, json={"id": "shared"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        async def send(wire):
            return (await client.post(wire.url)).json()

        sender = UploadSender(send, Gateway(client))
        ref = MediaRef(hash_name="local")
        settings = {"endpoint": "https://files"}
        owner = asyncio.create_task(service.upload(ref, settings, sender))
        await started.wait()
        waiter = asyncio.create_task(service.upload(ref, settings, sender))
        await asyncio.sleep(0)
        owner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await owner

        await service.close_clients([client])
        assert not client.is_closed
        release.set()
        assert (await waiter).value == "shared"
        await service.close_clients([client])
        assert client.is_closed

    asyncio.run(run())


def test_shared_task_has_its_own_timeout(tmp_path, image_bytes):
    async def run():
        service = MediaService(StateStore(tmp_path / "state.db"), upload_timeout=0.2)
        started = asyncio.Event()
        count = 0

        async def send(wire):
            nonlocal count
            count += 1
            started.set()
            await asyncio.Event().wait()

        settings = {"endpoint": "https://files"}
        ref = MediaRef(hash_name="local")
        owner = asyncio.create_task(service.upload(ref, settings, send, timeout=5))
        await started.wait()
        waiter = asyncio.create_task(service.upload(ref, settings, send, timeout=5))
        errors = await asyncio.gather(owner, waiter, return_exceptions=True)
        assert errors[0] is errors[1]
        assert isinstance(errors[0], TimeoutError)
        assert count == 1

    asyncio.run(run())


@pytest.mark.parametrize("difference", ["account", "hash_name", "database"])
def test_different_upload_identities_do_not_share_tasks(tmp_path, image_bytes, difference):
    async def run():
        service = MediaService(StateStore(tmp_path / "state.db"))
        other = MediaService(StateStore(tmp_path / ("other.db" if difference == "database" else "state.db")))
        settings = {"endpoint": "https://files"}
        other_settings = {"endpoint": "https://other"} if difference == "account" else settings
        both_started = asyncio.Event()
        count = 0

        async def send(wire):
            nonlocal count
            count += 1
            if count == 2:
                both_started.set()

            await both_started.wait()
            return {"id": "uploaded"}

        await asyncio.wait_for(asyncio.gather(
            service.upload(MediaRef(hash_name="first"), settings, send),
            other.upload(MediaRef(hash_name="second" if difference == "hash_name" else "first"), other_settings, send),
        ), 5)
        assert count == 2

    asyncio.run(run())


def test_external_process_lease_still_waits_for_persisted_result(tmp_path, image_bytes):
    from TIYA.llm.provider.files import FilesProvider

    async def run():
        state = StateStore(tmp_path / "state.db")
        service = MediaService(state)
        provider = FilesProvider({"endpoint": "https://files"})
        key = f"{provider.scope}:local"
        assert state.claim(key, "external", 10)

        async def forbidden(wire):
            pytest.fail("an external process already performs this upload")

        waiting = asyncio.create_task(service.upload(MediaRef(hash_name="local"), provider, forbidden))
        await asyncio.sleep(0.1)
        state.put("uploads", key, {"file_id": "external-file", "mime_type": "image/png"})
        state.release(key, "external")
        assert (await asyncio.wait_for(waiting, 5)).value == "external-file"

    asyncio.run(run())


def test_chat_owner_can_close_while_another_chat_finishes_shared_upload(tmp_path, monkeypatch):
    from llm_config_fixtures import candidate, configuration
    from TIYA.llm.chat import Chat
    from TIYA.llm.model import ImagePrompt

    started = asyncio.Event()
    joined = asyncio.Event()
    release = asyncio.Event()
    uploads = []
    models = []
    clients = []
    calls = 0
    real_upload = MediaService.upload
    real_client = httpx.AsyncClient
    output = BytesIO()
    Image.new("RGB", (2, 2)).save(output, "PNG")
    content = output.getvalue()

    async def upload(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            joined.set()

        return await real_upload(self, *args, **kwargs)

    async def handler(request):
        if request.url.path == "/files":
            uploads.append(request)
            assert request.extensions["timeout"]["read"] > 100
            started.set()
            await release.wait()
            return httpx.Response(200, json={"id": "shared"}, headers={"x-request-id": "file-platform"})

        models.append(request)
        return httpx.Response(200, json={"output": []}, headers={"x-request-id": "model-platform"})

    def client(**kwargs):
        instance = real_client(transport=httpx.MockTransport(handler))
        clients.append(instance)
        return instance

    monkeypatch.setattr(httpx, "AsyncClient", client)
    monkeypatch.setattr(MediaService, "upload", upload)

    async def run():
        config = configuration("main", [candidate(
            "responses_api:a:m", "responses_api", "a", "m", {"endpoint": "https://model/responses"},
            {"stream": False}, files_settings={"endpoint": "https://files/files"},
        )])
        first = Chat(config=config)
        second = Chat(config=config)
        owner = None
        waiter = None
        try:
            owner = asyncio.create_task(first.ask(ImagePrompt("user", content), no_wait=True, timeout=5))
            await asyncio.wait_for(started.wait(), 5)
            waiter = asyncio.create_task(second.ask(ImagePrompt("user", content), no_wait=True, timeout=5))
            await asyncio.wait_for(joined.wait(), 5)
            await asyncio.wait_for(first.close(), 1)
            assert await owner is None
            assert not clients[0].is_closed
            release.set()
            assert await waiter is not None
            rows = second._runtime.ledger.query_requests()
            file_rows = [row for row in rows if row["identity"]["operation"] == "files"]
            assert len(file_rows) == len(uploads) == 1
            assert file_rows[0]["status"] == "completed"
            assert file_rows[0]["diagnostic"]["platform_request_id"] == "file-platform"
            assert len(models) == 1

        finally:
            release.set()
            await first.close()
            await second.close()
            if owner is not None and waiter is not None:
                await asyncio.gather(owner, waiter, return_exceptions=True)

        assert all(instance.is_closed for instance in clients)

    asyncio.run(run())


def test_cancel_during_sqlite_claim_does_not_leave_an_orphan_lease(tmp_path, image_bytes, monkeypatch):
    import threading

    async def run():
        state = StateStore(tmp_path / "state.db")
        service = MediaService(state)
        entered = asyncio.Event()
        finished = asyncio.Event()
        unblock = threading.Event()
        loop = asyncio.get_running_loop()
        real_claim = state.claim

        def claim(*args):
            loop.call_soon_threadsafe(entered.set)
            assert unblock.wait(5)
            try:
                return real_claim(*args)

            finally:
                loop.call_soon_threadsafe(finished.set)

        async def forbidden(wire):
            pytest.fail("a cancelled task must not begin uploading")

        monkeypatch.setattr(state, "claim", claim)
        owner = asyncio.create_task(service.upload(MediaRef(hash_name="local"),
                                                   {"endpoint": "https://files"}, forbidden))
        await asyncio.wait_for(entered.wait(), 5)
        owner.cancel()
        await asyncio.sleep(0)
        unblock.set()
        with pytest.raises(asyncio.CancelledError):
            await owner

        await asyncio.wait_for(finished.wait(), 5)
        with state.db.transaction() as connection:
            assert connection.execute("SELECT count(*) FROM leases").fetchone()[0] == 0

    asyncio.run(run())


def test_new_waiter_waits_for_cancelled_upload_cleanup_before_starting(tmp_path, image_bytes):
    async def run():
        service = MediaService(StateStore(tmp_path / "state.db"))
        started = asyncio.Event()
        cleaning = asyncio.Event()
        finish_cleanup = asyncio.Event()

        async def old_send(wire):
            started.set()
            try:
                await asyncio.Event().wait()

            finally:
                cleaning.set()
                await finish_cleanup.wait()

        async def new_send(wire):
            return {"id": "new"}

        settings = {"endpoint": "https://files"}
        ref = MediaRef(hash_name="local")
        owner = asyncio.create_task(service.upload(ref, settings, old_send))
        await started.wait()
        owner.cancel()
        await cleaning.wait()
        replacement = asyncio.create_task(service.upload(ref, settings, new_send))
        await asyncio.sleep(0)
        assert not replacement.done()
        finish_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await owner

        assert (await replacement).value == "new"

    asyncio.run(run())
