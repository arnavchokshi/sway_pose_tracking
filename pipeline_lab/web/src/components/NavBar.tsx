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

  return (
    <div className="app-nav-outer">
      <header className="nav-shell">
        <div className="nav-shell__left">
          <div className="nav-brand">Sway Studio</div>

          <nav className="nav-links" aria-label="Primary">
            <NavLink to="/" end className={({ isActive }) => (isActive ? 'nav-link nav-link--active' : 'nav-link')}>
              Lab
            </NavLink>
            <NavLink to="/config" className={({ isActive }) => (isActive ? 'nav-link nav-link--active' : 'nav-link')}>
              Config Builder
            </NavLink>
            <NavLink to="/compare" className={({ isActive }) => (isActive ? 'nav-link nav-link--active' : 'nav-link')}>
              Compare
            </NavLink>
            <NavLink to="/scoring" className={({ isActive }) => (isActive ? 'nav-link nav-link--active' : 'nav-link')}>
              Scoring UI
            </NavLink>
          </nav>
        </div>

        <div className="nav-shell__right">
          {videoFile ? (
            <>
              <div className="nav-video-chip">
                <span className="nav-video-chip__dot" aria-hidden />
                <div className="nav-video-chip__text">
                  <span className="nav-video-chip__label">Shared video</span>
                  <span className="nav-video-chip__name" title={videoLabel}>
                    {videoLabel}
                  </span>
                </div>
              </div>
              <button type="button" className="btn btn--compact" onClick={openFilePicker}>
                Replace file
              </button>
            </>
          ) : (
            <button type="button" className="btn primary btn--compact" onClick={openFilePicker}>
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
