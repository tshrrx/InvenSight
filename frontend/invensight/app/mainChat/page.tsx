"use client";

import { useState, useRef, useEffect } from "react";
import Aurora from "../components/Aurora";
import { Upload, Send, X, Loader2, AlertCircle } from "lucide-react";
import { api } from "./apiservice";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

export default function MainPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [userId, setUserId] = useState<string>("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Generate or retrieve user ID
  useEffect(() => {
    let storedUserId = localStorage.getItem("invensight_user_id");
    if (!storedUserId) {
      storedUserId = `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      localStorage.setItem("invensight_user_id", storedUserId);
    }
    setUserId(storedUserId);
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (file.type !== "application/pdf") {
      setError("Please upload a PDF file");
      return;
    }

    setError(null);
    setIsUploading(true);

    try {
      // Upload to backend
      const response = await api.uploadPDF(userId, file);
      setUploadedFile(file);
      
      // Show uploading message
      const uploadingMessage: Message = {
        id: Date.now().toString(),
        role: "assistant",
        content: `📄 ${response.filename} uploaded! Processing... Please wait.`,
        timestamp: new Date(),
      };
      setMessages([uploadingMessage]);
      
      // Poll for completion
      let attempts = 0;
      const maxAttempts = 60; // 60 seconds max wait
      
      const checkStatus = setInterval(async () => {
        attempts++;
        const status = await api.checkUploadStatus(userId, file.name);
        
        if (status.status === "completed") {
          clearInterval(checkStatus);
          const readyMessage: Message = {
            id: (Date.now() + 1).toString(),
            role: "assistant",
            content: `✅ ${file.name} is ready! You can now ask questions about the document.`,
            timestamp: new Date(),
          };
          setMessages([readyMessage]);
          setIsUploading(false);
        } else if (status.status.startsWith("failed")) {
          clearInterval(checkStatus);
          setError(`Processing failed: ${status.status}`);
          setIsUploading(false);
        } else if (attempts >= maxAttempts) {
          clearInterval(checkStatus);
          setError("Processing is taking longer than expected. Please try querying anyway.");
          setIsUploading(false);
        }
      }, 1000); // Check every second
    
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
      setUploadedFile(null);
      setIsUploading(false);
    }
  };

  const removeFile = () => {
    setUploadedFile(null);
    setMessages([]);
    setInput("");
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleSendMessage = async () => {
    if (!uploadedFile) {
      setError("Please upload a PDF file first to start the conversation");
      return;
    }

    if (!input.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);
    setError(null);

    try {
      const response = await api.query(userId, input);
      
      // Check if it's a "processing" response
      if (response.answer.includes("still being processed")) {
        const retryMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: response.answer + "\n\n⏳ Retrying in 5 seconds...",
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, retryMessage]);
        
        // Auto-retry after 5 seconds
        setTimeout(() => {
          setInput(input); // Restore the query
          handleSendMessage(); // Retry
        }, 5000);
        
        return;
      }
      
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: response.answer,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Query failed");
      
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "Sorry, I encountered an error. Please try again in a moment.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <main className="relative flex min-h-screen w-full items-center justify-center font-sans bg-black dark:bg-black overflow-hidden">
      <style jsx global>{`
        /* Custom Scrollbar - only shows when needed */
        .custom-scrollbar {
          overflow-y: auto;
        }

        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }

        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }

        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.2);
          border-radius: 10px;
        }

        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(255, 255, 255, 0.3);
        }

        /* Remove scrollbar arrows/buttons */
        .custom-scrollbar::-webkit-scrollbar-button {
          display: none;
        }

        /* Firefox */
        .custom-scrollbar {
          scrollbar-width: thin;
          scrollbar-color: rgba(255, 255, 255, 0.2) transparent;
        }
      `}</style>

      <div className="w-full h-screen">
        <Aurora
          colorStops={["#0099FF", "#00E5FF", "#00FFB3"]}
          blend={0.8}
          amplitude={1}
          speed={0.5}
        />
        <div className="absolute inset-0 w-full h-full bg-transparent pointer-events-auto">
          <div className="relative flex flex-col w-full h-full max-w-5xl mx-auto p-4">
            {/* Header */}
            <div className="text-center py-6">
              <h1 className="text-3xl font-bold text-white mb-2">
                InvenSight AI Assistant
              </h1>
              <p className="text-gray-300 text-sm">
                Upload a PDF and start your conversation
              </p>
              {userId && (
                <p className="text-gray-500 text-xs mt-1">
                  Session ID: {userId.slice(0, 20)}...
                </p>
              )}
            </div>

            {/* Error Alert */}
            {error && (
              <div className="mb-4 bg-red-500/10 border border-red-500/50 rounded-lg p-3 flex items-center gap-2">
                <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
                <p className="text-red-300 text-sm">{error}</p>
                <button
                  onClick={() => setError(null)}
                  className="ml-auto text-red-400 hover:text-red-300"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            )}

            {/* Chat Container */}
            <div className="flex-1 overflow-hidden bg-white/5 backdrop-blur-lg rounded-2xl border border-white/10 shadow-2xl flex flex-col">
              {/* Messages Area */}
              <div className="flex-1 p-6 space-y-4 custom-scrollbar">
                {messages.length === 0 ? (
                  <div className="flex items-center justify-center h-full">
                    <div className="text-center text-gray-400">
                      {isUploading ? (
                        <>
                          <Loader2 className="w-16 h-16 mx-auto mb-4 opacity-50 animate-spin" />
                          <p className="text-lg">Uploading PDF...</p>
                        </>
                      ) : (
                        <>
                          <Upload className="w-16 h-16 mx-auto mb-4 opacity-50" />
                          <p className="text-lg">
                            {uploadedFile
                              ? "Start your conversation"
                              : "Upload a PDF to begin"}
                          </p>
                        </>
                      )}
                    </div>
                  </div>
                ) : (
                  messages.map((message) => (
                    <div
                      key={message.id}
                      className={`flex ${
                        message.role === "user"
                          ? "justify-end"
                          : "justify-start"
                      }`}
                    >
                      <div
                        className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                          message.role === "user"
                            ? "bg-blue-500 text-white"
                            : "bg-white/10 text-white border border-white/20"
                        }`}
                      >
                        <p className="text-sm whitespace-pre-wrap">
                          {message.content}
                        </p>
                        <p className="text-xs mt-1 opacity-70">
                          {message.timestamp.toLocaleTimeString()}
                        </p>
                      </div>
                    </div>
                  ))
                )}
                {isLoading && (
                  <div className="flex justify-start">
                    <div className="bg-white/10 text-white border border-white/20 rounded-2xl px-4 py-3">
                      <div className="flex space-x-2">
                        <div className="w-2 h-2 bg-white rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-white rounded-full animate-bounce delay-100"></div>
                        <div className="w-2 h-2 bg-white rounded-full animate-bounce delay-200"></div>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              {/* Input Area */}
              <div className="border-t border-white/10 p-4 bg-white/5">
                {/* Uploaded File Display */}
                {uploadedFile && (
                  <div className="mb-3 flex items-center justify-between bg-white/10 rounded-lg px-4 py-2">
                    <div className="flex items-center space-x-2">
                      <Upload className="w-4 h-4 text-blue-400" />
                      <span className="text-sm text-white">
                        {uploadedFile.name}
                      </span>
                      <span className="text-xs text-gray-400">
                        ({(uploadedFile.size / 1024).toFixed(2)} KB)
                      </span>
                    </div>
                    <button
                      onClick={removeFile}
                      className="text-red-400 hover:text-red-300 transition-colors"
                      title="Remove PDF and clear conversation"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                )}

                {/* Input Row */}
                <div className="flex items-end space-x-2">
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf"
                    onChange={handleFileUpload}
                    className="hidden"
                    id="pdf-upload"
                    disabled={isUploading}
                  />
                  <label
                    htmlFor="pdf-upload"
                    className={`cursor-pointer bg-white/10 hover:bg-white/20 transition-colors rounded-xl p-3 border border-white/20 ${
                      isUploading ? "opacity-50 cursor-not-allowed" : ""
                    }`}
                    title="Upload PDF"
                  >
                    {isUploading ? (
                      <Loader2 className="w-5 h-5 text-white animate-spin" />
                    ) : (
                      <Upload className="w-5 h-5 text-white" />
                    )}
                  </label>

                  <textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder={
                      uploadedFile
                        ? "Type your message..."
                        : "Upload a PDF first..."
                    }
                    disabled={!uploadedFile || isLoading}
                    className={`flex-1 bg-white/10 text-white placeholder-gray-400 rounded-xl px-4 py-3 border border-white/20 focus:outline-none focus:border-blue-400 resize-none max-h-32 custom-scrollbar ${
                      !uploadedFile || isLoading ? "opacity-50 cursor-not-allowed" : ""
                    }`}
                    rows={1}
                  />

                  <button
                    onClick={handleSendMessage}
                    disabled={!input.trim() || !uploadedFile || isLoading}
                    className="bg-blue-500 hover:bg-blue-600 disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors rounded-xl p-3"
                    title={!uploadedFile ? "Upload a PDF first" : "Send message"}
                  >
                    {isLoading ? (
                      <Loader2 className="w-5 h-5 text-white animate-spin" />
                    ) : (
                      <Send className="w-5 h-5 text-white" />
                    )}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}