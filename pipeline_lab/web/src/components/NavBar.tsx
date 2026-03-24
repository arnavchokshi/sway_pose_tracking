import { NavLink } from 'react-router-dom'
import { useRef } from 'react'
import { useLab } from '../context/LabContext'
import { isProbableVideoFile, VIDEO_ACCEPT_ATTR } from '../lib/videoFile'

export function NavBar() {
  const { videoFile, videoLabel, setVideo } = useLab()
  const fileRef = useRef<HTMLInputElement>(null)

  const onPickFile = (file?: File) => {
    if (!file) return
    if (!isProbableVideoFile(file)) {
      window.alert('Please choose a video file (MP4, MOV, WebM, MKV, AVI, or M4V).')
      return
    }
    setVideo(file)
  }

  const openFilePicker = () => {
    const el = fileRef.current
    if (!el) return
    el.value = ''
    el.click()
  }

  const navLinkStyle = ({ isActive }: { isActive: boolean }) => ({
    padding: '0.5rem 1rem',
    borderRadius: '999px',
    fontSize: '0.85rem',
    fontWeight: isActive ? 500 : 400,
    color: isActive ? '#fff' : 'var(--text-muted)',
    background: isActive ? 'rgba(255,255,255,0.08)' : 'transparent',
    textDecoration: 'none',
    transition: 'all 0.2s ease',
  })

  return (
    <div style={{ padding: '1rem 2rem', position: 'sticky', top: 0, zIndex: 50, marginBottom: '-1rem' }}>
      <header
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0.4rem 0.4rem',
          background: 'rgba(15, 23, 42, 0.4)',
          backdropFilter: 'blur(24px)',
          WebkitBackdropFilter: 'blur(24px)',
          border: '1px solid var(--glass-border)',
          borderRadius: '999px',
          boxShadow: '0 4px 24px -1px rgba(0,0,0,0.3)'
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
          <div 
            style={{ 
              fontWeight: 600, 
              fontSize: '0.9rem', 
              color: 'var(--halo-cyan)', 
              padding: '0.5rem 1.2rem',
              background: 'rgba(6, 182, 212, 0.1)',
              border: '1px solid rgba(6, 182, 212, 0.3)',
              borderRadius: '999px',
              marginLeft: '0.2rem'
            }}
          >
            Sway Studio
          </div>
          
          <div style={{ display: 'flex', gap: '0.25rem' }}>
            <NavLink to="/" end style={navLinkStyle}>
              Lab
            </NavLink>
            <NavLink to="/config" style={navLinkStyle}>
              Config Builder
            </NavLink>
            <NavLink to="/compare" style={navLinkStyle}>
              Compare
            </NavLink>
          </div>
        </div>

        <div style={{ position: 'relative', display: 'flex', alignItems: 'center', gap: '0.5rem', marginRight: '0.25rem' }}>
          {videoFile ? (
            <>
              <div style={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: '0.75rem',
                padding: '0.35rem 1rem 0.35rem 0.5rem',
                background: 'rgba(0,0,0,0.3)',
                border: '1px solid var(--glass-border)',
                borderRadius: '999px'
              }}>
                <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#10b981', boxShadow: '0 0 8px #10b981' }} />
                <div style={{ display: 'flex', flexDirection: 'column', lineHeight: 1.1 }}>
                  <span style={{ fontSize: '0.55rem', fontWeight: 700, letterSpacing: '0.06em', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Shared Video</span>
                  <span style={{ fontSize: '0.85rem', fontWeight: 500, color: '#fff', maxWidth: '160px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {videoLabel}
                  </span>
                </div>
              </div>
              <button 
                type="button" 
                className="btn" 
                style={{ padding: '0.5rem 1rem', borderRadius: '999px', fontSize: '0.8rem' }} 
                onClick={openFilePicker}
              >
                Replace file
              </button>
            </>
          ) : (
            <button 
              type="button" 
              className="btn primary" 
              style={{ padding: '0.5rem 1.25rem', borderRadius: '999px', fontSize: '0.85rem' }} 
              onClick={openFilePicker}
            >
              Upload video
            </button>
          )}
          <input
            ref={fileRef}
            type="file"
            accept={VIDEO_ACCEPT_ATTR}
            tabIndex={-1}
            className="sr-only-file-input"
            onChange={(e) => {
              onPickFile(e.target.files?.[0])
              e.target.value = ''
            }}
          />
        </div>
      </header>
    </div>
  )
}
