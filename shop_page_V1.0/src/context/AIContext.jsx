import React, { createContext, useContext, useState } from "react";

const AIContext = createContext();

export const AIProvider = ({ children }) => {
  const [selectedProducts, setSelectedProducts] = useState(() => {
    const stored = localStorage.getItem("selectedProducts");
    return stored ? JSON.parse(stored) : [];
  });
  

  const addProduct = (product) => {
    setSelectedProducts(prev => {
      if (prev.some(p => p.id === product.id)) return prev;
      const updated = [...prev, product];
      localStorage.setItem("selectedProducts", JSON.stringify(updated));
      return updated;
    });
  };

  const clearProducts = () => {
    setSelectedProducts([]);
    localStorage.removeItem("selectedProducts");
  };
  

  return (
    <AIContext.Provider value={{ selectedProducts, addProduct, clearProducts }}>
      {children}
    </AIContext.Provider>
  );
};

export const useAIContext = () => useContext(AIContext);
