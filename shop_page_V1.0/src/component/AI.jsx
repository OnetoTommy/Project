import React, { useState, useEffect } from "react";
import {
  MainContainer,
  ChatContainer,
  MessageList,
  Message,
  MessageInput,
  TypingIndicator,
} from "@chatscope/chat-ui-kit-react";
import { useOpenAI } from "./open_ai";
import { useAIContext } from "../context/AIContext";
import "../style/AI.css";
import "@chatscope/chat-ui-kit-styles/dist/default/styles.min.css";

function AI() {
  const { selectedProducts, clearProducts } = useAIContext();

  const [messages, setMessages] = useState([
    {
      message: "Hello, I'm ChatGPT!",
      sender: "ChatGPT",
      direction: "incoming",
    },
  ]);
  const [inputText, setInputText] = useState(""); // bind to input box
  const { typing, handleSend: sendToChatGPT } = useOpenAI(messages, setMessages);

  useEffect(() => {
    console.log("[AI.jsx useEffect] Triggered with selectedProducts:", selectedProducts);

    if (selectedProducts.length > 0) {
      const summary = selectedProducts.map((p, i) =>
        `Product ${i + 1}:\nTitle: ${p.title}\nPrice: $${p.price}\nRating: ${p.rating?.rate ?? 'N/A'}\nDescription: ${p.description}`
      ).join("\n\n");

      const prefill = `Please analyze the following products and give scores for the following products:\n\n${summary}`;
      console.log("[AI.jsx] Generated prefill:", prefill);

      setInputText(prefill); // set input content visibly
    }
  }, [selectedProducts]);

  const handleSend = (text) => {
    console.log("[AI.jsx] Sending message:", text);
    sendToChatGPT(text);
    setInputText("");      // clear after send
    clearProducts();       // clear selected
  };

  return (
    <div className="AI">
      <div className = "AI-container">
        <MainContainer>
          <ChatContainer>
            <MessageList>
              {typing && <TypingIndicator content="ChatGPT is typing..." />}
              {messages.map((message, i) => (
                <Message key={i} model={message} />
              ))}
            </MessageList>
            <MessageInput
              value={inputText}
              onChange={val => setInputText(val)}
              onSend={handleSend}
              placeholder="Type message here"
              attachButton={true}
            />
          </ChatContainer>
        </MainContainer>
      </div>
    </div>
  );
}

export default AI;  