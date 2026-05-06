import { useCallback, useState, type ReactNode } from 'react'

const HEADER_LOGO = '/g10x-header-logo.svg'

/** Fixed header bar + shadow; keep in sync with `ba-layout.css` scroll-margin. */
const HEADER_SCROLL_OFFSET_PX = 100

type SiteHeaderProps = {
  subtitle?: string
  /** e.g. “View clarifications” — rendered in the top-right before section nav */
  actions?: ReactNode
}

export function SiteHeader({ subtitle, actions }: SiteHeaderProps) {
  const [menuOpen, setMenuOpen] = useState(false)

  const scrollTo = useCallback((id: string) => {
    const el = document.getElementById(id)
    if (!el) return
    const top = el.getBoundingClientRect().top + window.scrollY - HEADER_SCROLL_OFFSET_PX
    window.scrollTo({ top: Math.max(0, top), behavior: 'smooth' })
    setMenuOpen(false)
  }, [])

  return (
    <header className="header-container visible">
      <div className="header-content">
        <div className="header-left">
          <div className="logo-wrapper ba-header-logo-wrap">
            <img className="logo-img" src={HEADER_LOGO} alt="G10X" width={52} height={69} />
          </div>
          <div className="ba-header-titles">
            <h1 className="header-title">AI Business Analyst</h1>
            {subtitle ? <p className="ba-header-subline hint-muted">{subtitle}</p> : null}
          </div>
          <button
            type="button"
            className="mobile-menu-toggle"
            aria-expanded={menuOpen}
            aria-controls="ba-mobile-nav"
            onClick={() => setMenuOpen((o) => !o)}
          >
            {menuOpen ? (
              <span className="close-icon" aria-hidden>
                <span />
                <span />
              </span>
            ) : (
              <span className="hamburger-icon" aria-hidden>
                <span />
                <span />
                <span />
              </span>
            )}
          </button>
        </div>
        <div className="header-right">
          {actions ? <div className="ba-header-actions">{actions}</div> : null}
          <nav className="nav-menu" aria-label="Page sections">
            <button type="button" className="nav-item" onClick={() => scrollTo('ba-input')}>
              Input
            </button>
            <button type="button" className="nav-item" onClick={() => scrollTo('ba-results')}>
              Results
            </button>
          </nav>
        </div>
      </div>
      <div
        id="ba-mobile-nav"
        className={`mobile-menu${menuOpen ? ' open' : ''}`}
        role="dialog"
        aria-modal="true"
        aria-label="Navigation menu"
      >
        <nav className="mobile-nav-menu" aria-label="Mobile sections">
          <p className="header-title-mobile">Sections</p>
          <button type="button" className="mobile-nav-item" onClick={() => scrollTo('ba-input')}>
            Input
          </button>
          <button type="button" className="mobile-nav-item" onClick={() => scrollTo('ba-results')}>
            Results
          </button>
        </nav>
      </div>
    </header>
  )
}
