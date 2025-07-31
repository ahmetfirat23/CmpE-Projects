import React from 'react'
import ReactDOM from 'react-dom/client'
import { RouterProvider } from 'react-router-dom'
import { typesafeBrowserRouter } from 'react-router-typesafe';
import { Home, DBManager, Coach, Jury, Player } from './pages/index'
import './index.css'

const { router, href } = typesafeBrowserRouter([
  {
    path: "/",
    Component: Home
  },
  {
    path: "/dbmanager",
    Component: DBManager
  },
  {
    path: "/coach",
    Component: Coach
  },
  {
    path: "/jury",
    Component: Jury
  },
  {
    path: "/player",
    Component: Player
  }
]);

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
);