import React, { useState, useEffect } from 'react'
import Skeleton from 'react-loading-skeleton';
import 'react-loading-skeleton/dist/skeleton.css';
import { NavLink } from 'react-router-dom';
import { useAIContext } from '../context/AIContext';
import { useNavigate } from 'react-router-dom';
import { toast } from 'react-toastify';
import "../style/Products.css"

const Products = () => {

  const [data, setData] = useState([]);
  const [filter, setFilter] = useState(data);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let componentMounted = true;

    const getProducts = async () => {
      setLoading(true);
      const response = await fetch("https://fakestoreapi.com/products");
      if (componentMounted) {
        setData(await response.clone().json());
        setFilter(await response.json());
        setLoading(false);
        console.log(filter);
      }
    };
    getProducts();
    return () => {
      componentMounted = false;
    }



  }, []);


  const Loading = () => {
    return (
      <>
        <div className="col-md-3">
          <Skeleton height={350} />
        </div>
        <div className="col-md-3">
          <Skeleton height={350} />
        </div>
        <div className="col-md-3">
          <Skeleton height={350} />
        </div>
        <div className="col-md-3">
          <Skeleton height={350} />
        </div>
      </>
    );
  };

  const filterProduct = (cat) => {
    const updatedList = data.filter((x) => x.category === cat);
    setFilter(updatedList);
  };

  const navigate = useNavigate();
  const { addProduct } = useAIContext();
  const handleAI = (product) => {
    console.log("[AddProduct] Adding:", product);
    addProduct(product);
    toast.success(`“${product.title}” Add Successfully`);
  };

  const ShowProducts = () => {
    return (
      <>
        <div className="buttons d-flex justify-content-center mb-3 pb-3">
          <button className="btn btn-outline-dark" onClick={() => setFilter(data)}>
            All
          </button>
          <button className="btn btn-outline-dark" onClick={() => filterProduct("men's clothing")}>
            Men's Clothing
          </button>
          <button className="btn btn-outline-dark" onClick={() => filterProduct("women's clothing")}>
            Women's Clothing
          </button>
          <button className="btn btn-outline-dark" onClick={() => filterProduct("jewelery")}>
            Jewelery
          </button>
          <button className="btn btn-outline-dark" onClick={() => filterProduct("electronics")}>
            Electronic
          </button>
        </div>

        {filter.map((product) => {
          return (
            <>
              <div className="col-md-3 d-flex">
                <div className="card h-100 w-100 d-flex flex-column justify-content-between shadow-sm">
                  <img
                    src={product.image}
                    className="card-img-top product-image"
                    alt={product.title}
                    style={{ height: "250px", objectFit: "contain" }}
                  />
                  <div className="card-body d-flex flex-column justify-content-between">
                    <h5 className="card-title">{product.title}</h5>
                    <p className="card-text fw-bold mb-2">${product.price}</p>
                  </div>
                  <div className="card-footer bg-white border-0">
                    <div style={{ display: "flex", gap: "1.2rem" }}>
                      <NavLink
                        to={`/products/${product.id}`}
                        className="btn btn-primary flex-fill w-50"
                      >
                        Buy Now
                      </NavLink>
                      <button
                        className="btn btn-primary flex-fill w-50"
                        onClick={() => handleAI(product)}
                      >
                        AI
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              {/* <div className="col-md-3">
                <div className="card h-100 d-flex flex-column">
                  <img
                    src={product.image}
                    className="card-img-top product-image"
                    alt={product.title}
                  />
                  <div className="card-body">
                    <h5 className="card-title">{product.title}</h5>
                    <p className="card-text">
                      ${product.price}
                    </p>
                    <NavLink to={`/products/${product.id}`} className="btn btn-primary me-2">
                      Buy Now
                    </NavLink>
                    <button className="btn btn-primary me-2" onClick={() => handleAI(product)}>AI</button>
                  </div>
                </div>
              </div> */}
            </>
          )
        })}
      </>

    )

  }


  return (
    <div>
      <div className="container my-5 py-2">
        <div className="row">
          <div className="col-12 mb-2">
            <h1 className='display-6 fw-bolder text-center'>Latest Products</h1>
            <hr />
          </div>
        </div>
        <div className='row justify-content-center g-4'>
          {loading ? <Loading /> : <ShowProducts />}
        </div>
      </div>

    </div>

  )
}

export default Products;