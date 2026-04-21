const FOOTER_LOGO = '/g10x-footer-logo.svg'

export function SiteFooter() {
  const year = new Date().getFullYear()
  return (
    <footer className="footer-container">
      <div className="footer-content">
        <div className="footer-logo-wrapper" aria-hidden>
          <img
            className="footer-logo-img"
            src={FOOTER_LOGO}
            alt=""
            width={32}
            height={43}
          />
        </div>
        <p className="footer-copyright">
          © {year} G10X · AI Business Analyst
        </p>
      </div>
    </footer>
  )
}
