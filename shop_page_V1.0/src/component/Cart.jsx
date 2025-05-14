import React from 'react';
import { useSelector } from 'react-redux';

const Cart = () => {
  // Access the cart state from Redux store
  const cartItems = useSelector((state) => state.handleCart);
  console.log("Redux cartItems from store:", cartItems);
  const totalItems = cartItems.reduce((sum, item) => sum + item.qty, 0); // calculate total quantity
  
  return (
    <div className="container py-5">
      <h2>Shopping Cart</h2>
      <div className="row">
        {cartItems.length > 0 ? (
          cartItems.map((product) => (
            <div className="col-md-4" key={product.id}>
              <div className="card">
                <img
                  src={product.image}
                  alt={product.title}
                  className="card-img-top"
                  height="340px"
                />
                <div className="card-body">
                  <h5 className="card-title">{product.title}</h5>
                  <p className="card-text">
                    Quantity: {product.qty} <br />
                    Price: ${product.qty * product.price}
                  </p>
                </div>
              </div>
            </div>
          ))
        ) : (
          <p>Your cart is empty.</p>
        )}
      </div>

      <div className="mt-4">
        <h3>Total Quantity: {totalItems}</h3>
      </div>
    </div>
  );
};

export default Cart;


// import React from "react";
// import { useDispatch } from "react-redux";
// import { deleteCart, addCart } from "../redux/action";

// const CartItem = ({ product }) => {
//   const dispatch = useDispatch();

//   // Handle button click to increase or decrease the quantity
//   const handleButton = (product, type) => {
//     if (type === "increase") {
//       dispatch(addCart(product));
//     } else if (type === "decrease") {
//       dispatch(deleteCart(product));
//     }
//   };

//   return (
//     <div className="row my-3">
//       <div className="col-md-4">
//         <img
//           src={product.image}
//           alt={product.title}
//           style={{ height: "200px", width: "180px" }}
//         />
//       </div>
//       <div className="col-md-4">
//         <h3>{product.title}</h3>
//         <p className="lead fw-bolder">
//           {product.qty} X ${product.price} = ${product.qty * product.price}
//         </p>
//         <button
//           className="btn btn-outline-dark me-4"
//           onClick={() => handleButton(product, "decrease")}
//         >
//           <i className="fa fa-minus"></i>
//         </button>
//         <button
//           className="btn btn-outline-dark"
//           onClick={() => handleButton(product, "increase")}
//         >
//           <i className="fa fa-plus"></i>
//         </button>
//       </div>
//     </div>
//   );
// };

// export default CartItem;

