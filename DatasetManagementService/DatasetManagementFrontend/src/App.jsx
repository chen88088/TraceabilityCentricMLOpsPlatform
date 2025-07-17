import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import AppRouter from "./router";

import React from "react";

export default function App() {
  return (
    // <div>
    //   <header>
    //     <h1>Dataset Management Service</h1>
    //   </header>
    //   <main>
    //     <AppRouter />
    //   </main>
    // </div>
    <div>
      <AppRouter />
    </div>
  );
}