import { useState, useEffect } from 'react'
import './HomePage.css'
import { Link } from 'react-router-dom'
import { SwayPhysicsCanvas } from '../components/SwayPhysicsBackground/SwayPhysicsCanvas'

export function HomePage() {
  const [color1, setColor1] = useState('#6366f1')
  const [color2, setColor2] = useState('#22d3ee')
  
  const [activeColor1, setActiveColor1] = useState('#6366f1')
  const [activeColor2, setActiveColor2] = useState('#22d3ee')


  // Use a debounced effect to safely commit logic without freezing UI
  useEffect(() => {
    const handler = setTimeout(() => {
      setActiveColor1(color1)
      setActiveColor2(color2)
    }, 150)
    return () => clearTimeout(handler)
  }, [color1, color2])

  return (
    <div className="home-container">
      {/* 3D WebGPU Background */}
      <SwayPhysicsCanvas 
        key={`${activeColor1}-${activeColor2}`} 
        color1={activeColor1} 
        color2={activeColor2} 
      />

      {/* Stage lighting overlays */}
      <div className="stage-glow stage-glow--left"></div>
      <div className="stage-glow stage-glow--right"></div>
      <div className="stage-glow stage-glow--top"></div>
      <div className="home-overlay"></div>

      {/* Top navigation */}
      <header className="home-top-bar">
        <div className="home-logo">
          <span className="logo-icon">✦</span>
          <span className="logo-text">SWAY</span>
        </div>
        <nav className="home-nav">
          <Link to="/lab">How It Works</Link>
          <Link to="/lab">Features</Link>
          <Link to="/compare">Compare</Link>
          <Link to="/optuna-sweep">Sweep</Link>
        </nav>
        <Link to="/lab" className="login-btn">Log In</Link>
      </header>

      {/* Center hero */}
      <div className="home-hero">
        <div className="hero-badge">
          <span className="badge-sparkle">✦</span>
          Smarter formations, clearer rehearsals
        </div>

        <h1 className="hero-headline">
          Design. <span className="gradient-word">Teach.</span> <span className="gradient-word-alt">Perfect.</span>
        </h1>

        <p className="hero-sub">
          The formation tool that syncs instantly between captains and dancers.<br />
          Create complex choreography, teach it effortlessly.
        </p>

        <Link to="/lab" className="hero-cta">
          <span>Get Started</span>
          <span className="cta-arrow">→</span>
        </Link>
      </div>

      {/* Floating confetti / particles */}
      <div className="confetti-field">
        {Array.from({ length: 18 }).map((_, i) => (
          <div
            key={i}
            className={`confetti confetti--${i % 4 === 0 ? 'square' : i % 4 === 1 ? 'diamond' : i % 4 === 2 ? 'rect' : 'circle'}`}
            style={{
              left: `${5 + Math.random() * 90}%`,
              top: `${5 + Math.random() * 70}%`,
              animationDelay: `${Math.random() * 6}s`,
              animationDuration: `${4 + Math.random() * 5}s`,
              opacity: 0.15 + Math.random() * 0.35,
            }}
          />
        ))}
      </div>

      {/* Bottom gradient fade */}
      <div className="bottom-fade"></div>

      {/* Aesthetics UI */}
      <div className="perf-controls">
        <label>
          <span className="perf-label">Primary Glow</span>
          <div className="color-picker-wrap">
            <input 
              type="color" 
              value={color1} 
              onChange={(e) => setColor1(e.target.value)}
            />
            <span className="color-hex">{color1.toUpperCase()}</span>
          </div>
        </label>
        <label>
          <span className="perf-label">Secondary Glow</span>
          <div className="color-picker-wrap">
            <input 
              type="color" 
              value={color2} 
              onChange={(e) => setColor2(e.target.value)}
            />
            <span className="color-hex">{color2.toUpperCase()}</span>
          </div>
        </label>
      </div>
    </div>
  )
}
