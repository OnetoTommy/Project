const cart = [];

const handleCart = (state = cart, action) => {
  console.log("Reducer called with action:", action);
  const product = action.payload;

  switch (action.type) {
    case "ADDITEM":
      console.log("Adding to cart:", product);
      const exist = state.find((x) => x.id === product.id);
      if (exist) {
        console.log("Already in cart, increasing qty");
        return state.map((x) =>
          x.id === product.id ? { ...x, qty: x.qty + 1 } : x
        );
      } else {
        console.log("New product, adding with qty 1");
        return [...state, { ...product, qty: 1 }];
      }


    case "DELITEM":
      const exist1 = state.find((x) => x.id === product.id);
      if (exist1.qty === 1) {
        return state.filter((x) => x.id !== exist1.id);
      } else {
        return state.map((x) =>
          x.id === product.id ? { ...x, qty: x.qty - 1 } : x
        );
      }

    default:
      return state;
  }
};

export default handleCart;
