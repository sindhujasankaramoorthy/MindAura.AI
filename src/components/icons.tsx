type P = { className?: string; size?: number };

const base = (size = 22) => ({
  width: size,
  height: size,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.6,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
});

export const NotebookIcon = ({ className, size }: P) => (
  <svg {...base(size)} className={className}>
    <path d="M6 3h11a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1Z" />
    <path d="M9 3v18M12.5 8.5h3M12.5 12h3" />
  </svg>
);

export const MicIcon = ({ className, size }: P) => (
  <svg {...base(size)} className={className}>
    <rect x="9" y="2.5" width="6" height="11" rx="3" />
    <path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21" />
  </svg>
);

export const CameraIcon = ({ className, size }: P) => (
  <svg {...base(size)} className={className}>
    <path d="M4 7h3l1.5-2h7L17 7h3a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V8a1 1 0 0 1 1-1Z" />
    <circle cx="12" cy="13" r="3.5" />
  </svg>
);

export const LeafIcon = ({ className, size }: P) => (
  <svg {...base(size)} className={className}>
    <path d="M20 4c0 9-5.5 14.5-14 15 0-9 5.5-14.5 14-15Z" />
    <path d="M6 19c2.5-4.5 6-8 10-10.5" />
  </svg>
);

export const CheckIcon = ({ className, size }: P) => (
  <svg {...base(size)} className={className}>
    <path d="m5 12.5 4.5 4.5L19 7" />
  </svg>
);

export const SparkIcon = ({ className, size }: P) => (
  <svg {...base(size)} className={className}>
    <path d="M12 3.5 13.6 9 19 10.5 13.6 12 12 17.5 10.4 12 5 10.5 10.4 9 12 3.5Z" />
    <path d="M18.5 16.5 19 18.5l2 .5-2 .5-.5 2-.5-2-2-.5 2-.5.5-2Z" />
  </svg>
);

export const ClockIcon = ({ className, size }: P) => (
  <svg {...base(size)} className={className}>
    <circle cx="12" cy="12" r="8.5" />
    <path d="M12 7.5V12l3 1.8" />
  </svg>
);
