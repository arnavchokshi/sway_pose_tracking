import { Link } from 'react-router-dom'
import './HomePage.css'
import { SwayPhysicsCanvas } from '../components/SwayPhysicsBackground/SwayPhysicsCanvas'

export function HomePage() {
  return (
    <div className="home-container">
      <SwayPhysicsCanvas color1="#1144ff" color2="#ffffff" noiseAmplitude={25} attractorStrength={40} cohesion={30} />
      <div className="overlay"></div>

      <div className="corner corner--tl"></div>
      <div className="corner corner--tr"></div>
      <div className="corner corner--bl"></div>
      <div className="corner corner--br"></div>

      <header className="top-bar">
        <div className="logo">Nexus Lab</div>
        <ul className="nav">
          <li><Link to="/lab">Research</Link></li>
          <li><Link to="/compare">Protocol</Link></li>
          <li><Link to="/optuna-sweep">Deploy</Link></li>
        </ul>
      </header>

      <div className="status-badge">
        <div className="status-dot"></div>
        <span className="status-text">System Active</span>
      </div>

      <div className="hero">
        <div className="hero-label">Introducing the next frontier</div>
        <div className="hero-word">Singularity</div>
        <div className="hero-sub">
          Where intelligence converges.
          <br />
          Beyond the threshold.
        </div>
        <div className="hero-line"></div>
      </div>

      <div className="panel-left">
        <div className="vertical-label">Cohesion Protocol - Nexus Lab</div>
        <div className="coords">
          <span className="accent">NET.ONLINE</span>
          <span>NODE 0x7F3A</span>
          <span>EPOCH 2891</span>
          <span>DRIFT 0.002</span>
          <span className="accent">V.4.0.1</span>
        </div>
      </div>

      <div className="panel-right">
        <div className="stat">
          <div className="stat-value">49K</div>
          <div className="stat-label">Active nodes</div>
        </div>
        <div className="stat">
          <div className="stat-value">0.7ms</div>
          <div className="stat-label">Latency</div>
        </div>
        <div className="stat">
          <div className="stat-value">99.97</div>
          <div className="stat-label">Uptime %</div>
        </div>
      </div>

      <div className="bottom-bar">
        <span className="tag">Neural mesh</span>
        <span className="tag">Swarm intelligence</span>
        <span className="tag">Quantum bridge</span>
        <span className="tag">Singularity</span>
      </div>
    </div>
  )
}
