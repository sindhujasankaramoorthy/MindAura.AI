export function AuraLogo({ size = 36 }: { size?: number }) {
  return (
    <span
      className="relative inline-flex items-center justify-center"
      style={{ width: size, height: size }}
      aria-hidden="true"
    >
      <span className="absolute inset-0 rounded-full bg-primary/25 blur-md animate-aura" />
      <span className="absolute inset-[-6px] rounded-full border border-primary/25 animate-ripple" />
      <span className="relative flex items-center justify-center rounded-full bg-primary/90 text-primary-foreground" style={{ width: size * 0.66, height: size * 0.66 }}>
        <svg viewBox="0 0 24 24" width={size * 0.4} height={size * 0.4} fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="M20 5c0 8-5 13-13 14 0-8 5-13 13-14Z" />
          <path d="M7 19c2-4 5-7 9-9" />
        </svg>
      </span>
    </span>
  );
}

export function BrandMark() {
  return (
    <span className="flex items-center gap-2.5">
      <AuraLogo />
      <span className="font-display text-lg tracking-tight text-primary-deep">
        MindAura <span className="text-muted-foreground font-sans text-sm">AI</span>
      </span>
    </span>
  );
}
