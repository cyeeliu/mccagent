import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Suspense, lazy } from "react";
import type { ReactNode } from "react";
import { createBrowserRouter, NavLink, Outlet, RouterProvider } from "react-router-dom";
import { BatchDetailPage, BatchesPage } from "./features/runs/BatchesPage";
import { TracePage } from "./features/trace/TracePage";

const DashboardPage = lazy(() =>
  import("./features/dashboard/DashboardPage").then((mod) => ({ default: mod.DashboardPage })),
);
const ComparePage = lazy(() =>
  import("./features/compare/ComparePage").then((mod) => ({ default: mod.ComparePage })),
);

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 1, staleTime: 5000 } },
});

function Layout() {
  return (
    <div className="app">
      <nav className="topnav">
        <NavLink to="/" className="brand" end>
          agent-eval
        </NavLink>
        <div className="nav-links">
          <NavLink to="/" end className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>
            批次
          </NavLink>
          <NavLink to="/dashboard" className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>
            仪表盘
          </NavLink>
          <NavLink to="/compare" className={({ isActive }) => isActive ? "nav-link active" : "nav-link"}>
            对比
          </NavLink>
        </div>
      </nav>
      <Outlet />
    </div>
  );
}

function route(page: ReactNode) {
  return <Suspense fallback={<div className="loading">加载中…</div>}>{page}</Suspense>;
}

const router = createBrowserRouter([
  {
    element: <Layout />,
    children: [
      { path: "/", element: <BatchesPage /> },
      { path: "/batches/:batchId", element: <BatchDetailPage /> },
      { path: "/runs/:runId", element: <TracePage /> },
      { path: "/dashboard", element: route(<DashboardPage />) },
      { path: "/compare", element: route(<ComparePage />) },
    ],
  },
]);

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  );
}
