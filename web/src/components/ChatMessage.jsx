import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// Tin nhắn: trợ lý (icon ✨, render markdown) hoặc người dùng (bong bóng xám, phải).
export default function ChatMessage({ role, content, streaming, error }) {
  if (role === "user") {
    return (
      <div className="msg user">
        <div className="bubble">{content}</div>
      </div>
    );
  }
  return (
    <div className="msg assistant">
      <div className="spark">✦</div>
      <div className={"assistant-body" + (error ? " error" : "")}>
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{content || ""}</ReactMarkdown>
        {streaming && <span className="caret" />}
      </div>
    </div>
  );
}
