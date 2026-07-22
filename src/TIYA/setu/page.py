"""Static page used to control the temporary remote browser."""

REMOTE_LOGIN_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
  <meta name="referrer" content="no-referrer">
  <title>Pixiv 临时登录</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #11151b; color: #eef2f7; overflow: hidden; }
    header {
      height: 58px; padding: 9px 12px; display: flex; align-items: center;
      justify-content: space-between; gap: 10px; background: #1a2029;
    }
    #status { font-size: 13px; line-height: 1.3; }
    button {
      border: 0; border-radius: 7px; padding: 9px 12px;
      background: #d84b59; color: white; font-weight: 650;
    }
    #stage {
      height: calc(100vh - 58px); display: flex; align-items: center;
      justify-content: center; touch-action: none; background: #07090c;
    }
    #screen {
      display: block; max-width: 100%; max-height: 100%;
      width: auto; height: auto; user-select: none; -webkit-user-drag: none;
    }
    #keyboard {
      position: fixed; left: 50%; bottom: 0; width: 2px; height: 2px;
      opacity: 0; pointer-events: none;
    }
  </style>
</head>
<body>
  <header>
    <div id="status">正在连接服务器上的临时浏览器…<br>输入内容会经过本 TIYA 服务器。</div>
    <button id="cancel" type="button">取消</button>
  </header>
  <main id="stage"><img id="screen" alt="远程 Pixiv 登录页面"></main>
  <textarea id="keyboard" autocomplete="off" autocapitalize="none" spellcheck="false"></textarea>
  <script>
    const screen = document.getElementById("screen");
    const stage = document.getElementById("stage");
    const status = document.getElementById("status");
    const keyboard = document.getElementById("keyboard");
    const socketUrl = new URL(location.href);
    socketUrl.pathname = location.pathname.replace(/\\/$/, "") + "/ws";
    socketUrl.search = "";
    socketUrl.hash = "";
    socketUrl.protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const socket = new WebSocket(socketUrl);
    let remoteWidth = 412;
    let remoteHeight = 915;
    let pointer = null;
    let composing = false;
    let pendingComposition = "";
    let compositionTimer = null;

    function send(payload) {
      if (socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify(payload));
      }
    }
    function point(event) {
      const rect = screen.getBoundingClientRect();
      return {
        x: Math.max(0, Math.min(remoteWidth, (event.clientX - rect.left) * remoteWidth / rect.width)),
        y: Math.max(0, Math.min(remoteHeight, (event.clientY - rect.top) * remoteHeight / rect.height))
      };
    }
    socket.onopen = () => { status.firstChild.textContent = "已连接，请完成 Pixiv 官方登录。"; };
    socket.onclose = event => {
      status.firstChild.textContent = event.code === 4003
        ? "该登录会话已由另一位管理员领取。"
        : "连接已关闭，可刷新页面重连。";
    };
    socket.onmessage = event => {
      const message = JSON.parse(event.data);
      if (message.type === "frame") {
        remoteWidth = message.width;
        remoteHeight = message.height;
        screen.src = "data:image/jpeg;base64," + message.data;
      } else if (message.type === "complete") {
        status.textContent = "登录凭证已获取，可以关闭本页面。";
      } else if (message.type === "error") {
        status.textContent = message.message;
      } else if (message.type === "keyboard") {
        if (message.show) {
          keyboard.setAttribute("inputmode", message.inputMode || "text");
          keyboard.focus({ preventScroll: true });
        } else {
          keyboard.blur();
        }
      }
    };
    stage.addEventListener("pointerdown", event => {
      if (!screen.src || !event.isPrimary) return;
      event.preventDefault();
      const p = point(event);
      pointer = { ...p, moved: false, pointerId: event.pointerId };
      stage.setPointerCapture(event.pointerId);
    });
    stage.addEventListener("pointermove", event => {
      if (!pointer || pointer.pointerId !== event.pointerId) return;
      event.preventDefault();
      const p = point(event);
      const dx = p.x - pointer.x;
      const dy = p.y - pointer.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) {
        pointer.moved = true;
        send({ type: "scroll", x: p.x, y: p.y, deltaX: -dx, deltaY: -dy });
        pointer.x = p.x;
        pointer.y = p.y;
      }
    });
    stage.addEventListener("pointerup", event => {
      if (!pointer || pointer.pointerId !== event.pointerId) return;
      event.preventDefault();
      const p = point(event);
      if (!pointer.moved) {
        send({ type: "click", x: p.x, y: p.y });
      }
      pointer = null;
    });
    stage.addEventListener("pointercancel", event => {
      if (pointer && pointer.pointerId === event.pointerId) pointer = null;
    });
    keyboard.addEventListener("compositionstart", () => {
      composing = true;
      pendingComposition = "";
      if (compositionTimer !== null) clearTimeout(compositionTimer);
    });
    keyboard.addEventListener("compositionend", event => {
      composing = false;
      pendingComposition = event.data || keyboard.value;
      compositionTimer = setTimeout(() => {
        if (pendingComposition) {
          send({ type: "text", text: pendingComposition });
          pendingComposition = "";
          keyboard.value = "";
        }
      }, 0);
    });
    keyboard.addEventListener("input", event => {
      if (composing || event.isComposing) return;
      if (compositionTimer !== null) {
        clearTimeout(compositionTimer);
        compositionTimer = null;
      }
      const text = event.data || pendingComposition || keyboard.value;
      if (text) send({ type: "text", text });
      pendingComposition = "";
      keyboard.value = "";
    });
    keyboard.addEventListener("keydown", event => {
      if (event.isComposing) return;
      if (["Enter", "Backspace", "Delete", "Tab", "Escape",
           "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(event.key)) {
        event.preventDefault();
        send({ type: "key", key: event.key });
      }
    });
    document.getElementById("cancel").addEventListener("click", () => {
      send({ type: "cancel" });
      status.textContent = "正在取消登录…";
    });
  </script>
</body>
</html>
"""
