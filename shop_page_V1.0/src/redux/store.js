// import { configureStore } from '@reduxjs/toolkit';
// import rootReducers from './reducer';

// const store = configureStore({
//   reducer: rootReducers,
//   devTools: process.env.NODE_ENV !== 'production',
// });

// export default store;

import { configureStore } from '@reduxjs/toolkit';
import { persistStore, persistReducer } from 'redux-persist';
import storage from 'redux-persist/lib/storage'; 
import { combineReducers } from 'redux';
import handleCart from './reducer/handleCart';

const persistConfig = {
  key: 'root',
  storage,
};

const rootReducer = combineReducers({
  handleCart,
});

const persistedReducer = persistReducer(persistConfig, rootReducer);

const store = configureStore({
  reducer: persistedReducer,
  devTools: process.env.NODE_ENV !== 'production',
});

const persistor = persistStore(store);

export { store, persistor };

