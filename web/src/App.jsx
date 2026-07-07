import { useRef, useState } from "react";
import { askStream } from "./lib/sse.js";
import WireGrid from "./components/WireGrid.jsx";
import ChatMessage from "./components/ChatMessage.jsx";
import Composer from "./components/Composer.jsx";
import CitationPanel from "./components/CitationPanel.jsx";

const GREETING =
  "Chào bạn, tôi là **LEXgpt** — trợ lý AI hỗ trợ phân tích và trả lời câu hỏi về " +
  "pháp luật Việt Nam cho doanh nghiệp nhỏ và vừa. Hãy đặt câu hỏi pháp lý bất kỳ, " +
  "tôi sẽ trả lời kèm **trích dẫn văn bản** liên quan.";

export default function App() {
  const [messages, setMessages] = useState([{ role: "assistant", content: GREETING }]);
  const [citations, setCitations] = useState([]);
  const [panelOpen, setPanelOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const scrollRef = useRef(null);

  const scrollToEnd = () =>
    requestAnimationFrame(() => {
      const el = scrollRef.current;
      if (el) el.scrollTop = el.scrollHeight;
    });

  function handleSend(question) {
    if (busy) return;
    setMessages((m) => [
      ...m,
      { role: "user", content: question },
      { role: "assistant", content: "", streaming: true },
    ]);
    setCitations([]);
    setBusy(true);
    setStatus("Đang truy hồi văn bản liên quan…");
    scrollToEnd();

    const setLastAssistant = (updater) =>
      setMessages((m) => {
        const copy = [...m];
        for (let i = copy.length - 1; i >= 0; i--) {
          if (copy[i].role === "assistant") {
            copy[i] = { ...copy[i], ...updater(copy[i]) };
            break;
          }
        }
        return copy;
      });

    askStream(question, {
      onCitations: (payload) => {
        setCitations(payload.citations || []);
        if ((payload.citations || []).length) setPanelOpen(true);
        setStatus("Đang soạn câu trả lời…");
      },
      onToken: (delta) => {
        setLastAssistant((prev) => ({ content: prev.content + delta }));
        scrollToEnd();
      },
      onDone: (payload) => {
        setLastAssistant((prev) => ({
          content: payload.answer || prev.content,
          streaming: false,
        }));
        setBusy(false);
        setStatus("");
        scrollToEnd();
      },
      onError: (msg) => {
        setLastAssistant((prev) => ({
          content: prev.content || `⚠️ ${msg}`,
          streaming: false,
          error: true,
        }));
        setBusy(false);
        setStatus(`Lỗi: ${msg}`);
      },
    });
  }

  return (
    <div className="app">
      <WireGrid />

      <header className="topbar">
        <div className="brand">
          LE<span className="brand-x">X</span>centra
        </div>
        {!panelOpen && citations.length > 0 && (
          <button className="reopen" onClick={() => setPanelOpen(true)}>
            📖 Văn bản liên quan ({citations.length})
          </button>
        )}
      </header>

      <div className="body">
        <main className="chat">
          <div className="messages" ref={scrollRef}>
            {messages.map((m, i) => (
              <ChatMessage key={i} role={m.role} content={m.content}
                streaming={m.streaming} error={m.error} />
            ))}
            {status && <div className="status">{status}</div>}
          </div>
          <Composer onSend={handleSend} busy={busy} />
        </main>

        <CitationPanel
          open={panelOpen}
          citations={citations}
          onClose={() => setPanelOpen(false)}
        />
      </div>
    </div>
  );
}
