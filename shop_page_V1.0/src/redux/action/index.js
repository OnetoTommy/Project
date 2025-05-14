// Add Item to Cart

export const addCart = (product) => {
  return{
    type : "ADDITEM",
    payload : product

  }
}


// Delete Item to Cart

export const delCart = (product) => {
  return{
    type : "DELITEM",
    payload : product

  }
}

// export const addCart = (product) => {
//   return {
//     type: "ADDITEM",
//     payload: product,
//   };
// };

// // Update export to use delCart instead of deleteCart
// export const delCart = (product) => {
//   return {
//     type: "DELITEM",
//     payload: product,
//   };
// };
