
import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import Avatar from "../components/Avatar";
import MarkdownMessage from "../components/MarkdownMessage";
import { useNavigate } from "react-router-dom";

function Home() {
    const [messages, setMessages] = useState([
        { role: "assistant", content: "Hello! How can I help you today?" },
    ]);
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const chatEndRef = useRef<HTMLDivElement>(null);
    const navigate = useNavigate();

    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    const sendMessage = async () => {
        if (!input.trim()) return;
        const userMsg = { role: "user", content: input };
        setMessages((msgs) => [...msgs, userMsg]);
        setInput("");
        setLoading(true);
        try {
            const res = await axios.post("http://127.0.0.1:8000/chat", {
                message: input,
            });
            setMessages((msgs) => [
                ...msgs,
                { role: "assistant", content: res.data.response },
            ]);
        } catch (err) {
            setMessages((msgs) => [
                ...msgs,
                { role: "assistant", content: "Sorry, there was an error." },
            ]);
        }
        setLoading(false);
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === "Enter" && !loading) {
            sendMessage();
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-950 flex flex-col items-center justify-center py-8 px-2">
            <div className="mb-6 text-center">
                <h1 className="text-3xl md:text-4xl font-extrabold text-white drop-shadow-lg tracking-tight">
                    Agentic LLM Assistant
                </h1>
                <p className="text-slate-300 text-sm mt-1">Your local AI assistant with tools and memory</p>
            </div>
            <button
                className="mb-6 bg-gradient-to-r from-red-700 to-red-500 px-5 py-2 text-lg text-white rounded-lg shadow hover:from-red-600 hover:to-red-400 font-bold transition"
                onClick={() => navigate("/file-upload")}
            >
                Upload File
            </button>
            <div className="w-full max-w-2xl bg-white/10 rounded-xl shadow-2xl flex flex-col h-[80vh] border border-slate-700">
                <div className="flex-1 overflow-y-auto p-6 space-y-4">
                    {messages.map((msg, idx) => (
                        <div
                            key={idx}
                            className={`flex items-end gap-2 ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                        >
                            {msg.role === "assistant" && <Avatar role="assistant" />}
                            <div
                                className={`px-4 py-2 rounded-lg max-w-[75%] whitespace-pre-line text-base shadow-md ${
                                    msg.role === "user"
                                        ? "bg-blue-600 text-white rounded-br-none"
                                        : "bg-slate-200 text-gray-900 rounded-bl-none"
                                }`}
                            >
                                <MarkdownMessage content={msg.content} />
                            </div>
                            {msg.role === "user" && <Avatar role="user" />}
                        </div>
                    ))}
                    <div ref={chatEndRef} />
                </div>
                <div className="p-4 border-t border-slate-700 bg-white/20 flex items-center gap-2">
                    <input
                        type="text"
                        className="flex-1 px-4 py-2 rounded-lg border border-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-400 bg-white/80 text-gray-900"
                        placeholder="Type your message..."
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={handleKeyDown}
                        disabled={loading}
                        autoFocus
                    />
                    <button
                        onClick={sendMessage}
                        disabled={loading || !input.trim()}
                        className="bg-blue-600 hover:bg-blue-700 text-white px-5 py-2 rounded-lg font-semibold shadow transition disabled:opacity-50"
                    >
                        {loading ? "..." : "Send"}
                    </button>
                </div>
            </div>
        </div>
    );
}

export default Home;