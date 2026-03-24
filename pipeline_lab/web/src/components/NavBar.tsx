import { NavLink } from 'react-router-dom'
import { Settings, FlaskConical, SplitSquareHorizontal } from 'lucide-react'

export function NavBar() {
  return (
    <nav
      className="glass-panel"
      style={{
        display: 'flex',
        gap: '1rem',
        padding: '1rem 2rem',
        marginBottom: '2rem',
        alignItems: 'center',
        flexWrap: 'wrap',
      }}
    >
      <div style={{ fontWeight: 700, fontSize: '1.2rem', color: 'var(--halo-cyan)', marginRight: '2rem' }}>
        Sway Studio
      </div>
      <NavLink
        to="/"
        end
        className={({ isActive }) => `btn ${isActive ? 'primary' : ''}`}
        style={{ border: 'none', background: 'transparent' }}
      >
        <FlaskConical size={18} /> Lab
      </NavLink>
      <NavLink
        to="/config"
        className={({ isActive }) => `btn ${isActive ? 'primary' : ''}`}
        style={{ border: 'none', background: 'transparent' }}
      >
        <Settings size={18} /> Config builder
      </NavLink>
      <NavLink
        to="/compare"
        className={({ isActive }) => `btn ${isActive ? 'primary' : ''}`}
        style={{ border: 'none', background: 'transparent' }}
      >
        <SplitSquareHorizontal size={18} /> Compare
      </NavLink>
    </nav>
  )
}
