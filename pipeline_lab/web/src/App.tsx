import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { LabProvider } from './context/LabContext'
import { NavBar } from './components/NavBar'
import { LabPage } from './pages/LabPage'
import { ConfigPage } from './pages/ConfigPage'
import { ComparePage } from './pages/ComparePage'
import { WatchPage } from './pages/WatchPage'
import { LiveSandboxPage } from './pages/LiveSandboxPage'

export default function App() {
  return (
    <BrowserRouter>
      <LabProvider>
        <div className="app-root">
          <NavBar />
          <main className="main-content">
            <Routes>
              <Route path="/" element={<LabPage />} />
              <Route path="/config" element={<ConfigPage />} />
              <Route path="/compare" element={<ComparePage />} />
              <Route path="/watch/:id/live" element={<LiveSandboxPage />} />
              <Route path="/watch/:id" element={<WatchPage />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </main>
        </div>
      </LabProvider>
    </BrowserRouter>
  )
}
