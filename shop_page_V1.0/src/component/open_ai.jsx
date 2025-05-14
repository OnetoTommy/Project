import { useState } from 'react';

const API_KEY = "sk-lMvjvDGYTQp2wqhpNsqFiURGP3tV2MEJSoZNlORLp4PoiTCv";  

export function useOpenAI(messages, setMessages) {
  const [typing, setTyping] = useState(false);

  const handleSend = async (message) => {
    const newMessage = {
      message: message,
      sender: "user",
      direction: "outgoing"
    };
    const newMessages = [...messages, newMessage];
    setMessages(newMessages);
    setTyping(true);
    await processMessageToChatGPT(newMessages);
  };

  async function processMessageToChatGPT(chatMessages) {
    let apiMessages = chatMessages.map((messageObject) => {
      let role = "";
      if (messageObject.sender === "ChatGPT") {
        role = "assistant";
      } else {
        role = "user";
      }
      return { role: role, content: messageObject.message };
    });

    const systemMessage = {
      role: "system",
      content: "Explain all concepts like I'm 10 years old"
    };

    const apiRequestBody = {
      model: "gpt-3.5-turbo",
      messages: [
        systemMessage,
        ...apiMessages
      ]
    };

    await fetch("https://xiaoai.plus/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify(apiRequestBody)
    })
      .then((data) => data.json())
      .then((data) => {
        setMessages([
          ...chatMessages,
          {
            message: data.choices[0].message.content,
            sender: "ChatGPT",
            direction: "incoming"
          }
        ]);
        setTyping(false);
      })
      .catch((error) => {
        console.error("Error:", error);
        setTyping(false);
      });
  }

  async function sendImageForAnalysis(imageUrl) {
    const apiRequestBody = {
      model: "gpt-4o",
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "What's in this image?" },
            {
              type: "image_url",
              image_url: {
                url: imageUrl,
              }
            }
          ],
        }
      ],
      max_tokens: 300,
    };

    await fetch("https://xiaoai.plus/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify(apiRequestBody)
    })
      .then((data) => data.json())
      .then((data) => {
        setMessages([
          ...messages,
          {
            message: data.choices[0].message.content,
            sender: "ChatGPT",
            direction: "incoming"
          }
        ]);
      })
      .catch((error) => {
        console.error("Error sending image:", error);
      });
  }

  async function generateEmbeddings(text) {
    const apiRequestBody = {
      model: "text-embedding-ada-002",
      input: text,
      encoding_format: "float"
    };

    await fetch("https://xiaoai.plus/v1/embeddings", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify(apiRequestBody)
    })
      .then((data) => data.json())
      .then((data) => {
        console.log("Generated Embeddings:", data);
        setMessages([
          ...messages,
          {
            message: `Generated embeddings for text: ${JSON.stringify(data)}`,
            sender: "ChatGPT",
            direction: "incoming"
          }
        ]);
      })
      .catch((error) => {
        console.error("Error generating embeddings:", error);
      });
  }

  return { typing, handleSend };
}
