import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css'; // Import your main CSS file (where Tailwind directives are)
import App from './App'; // Import your App component

// Create a root to render your React application
const root = ReactDOM.createRoot(document.getElementById('root'));

// Render the App component into the root
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
// import reportWebVitals from './reportWebVitals';
// reportWebVitals();
