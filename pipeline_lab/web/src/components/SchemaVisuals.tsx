import type { ReactNode } from 'react'

export function SchemaFieldVisual({ fieldId, value }: { fieldId: string; value?: any }): ReactNode {
  const SvgBox = ({ children, viewBox = '0 0 240 80' }: { children: ReactNode; viewBox?: string }) => (
    <div style={{ padding: '0.4rem 0', background: 'rgba(0,0,0,0.25)', borderTop: '1px solid rgba(255,255,255,0.05)', borderBottom: '1px solid rgba(255,255,255,0.05)', marginBottom: '0.35rem', marginTop: '0.1rem' }}>
      <svg viewBox={viewBox} style={{ width: '100%', height: 'auto', display: 'block', maxWidth: 360, margin: '0 auto', maxHeight: 76 }}>
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--text-muted)" />
          </marker>
        </defs>
        {children}
      </svg>
    </div>
  )

  const num = typeof value === 'number' ? value : 0;
  
  switch (fieldId) {
    case 'sway_pretrack_nms_iou': {
      const overlapX = 90 - (num * 40); // 0.1 -> 86, 0.9 -> 54
      return (
        <SvgBox>
          <rect x="50" y="15" width="40" height="50" stroke="var(--halo-cyan)" fill="rgba(6,182,212,0.1)" strokeWidth="2" rx="4" />
          <rect x={overlapX} y="20" width="40" height="50" stroke="#f87171" fill="none" strokeWidth="2" strokeDasharray="4 4" rx="4" opacity="0.7" />
          <path d="M 135 40 L 155 40" stroke="var(--text-muted)" strokeWidth="2" markerEnd="url(#arrow)" />
          <rect x="170" y="15" width="40" height="50" stroke="var(--halo-cyan)" fill="rgba(6,182,212,0.25)" strokeWidth="3" rx="4" />
          <text x="80" y="10" fill="var(--text-muted)" fontSize="9" textAnchor="middle">Overlap: {Math.round(num * 100)}%</text>
        </SvgBox>
      )
    }
    case 'sway_yolo_conf': {
      const thresholdY = 65 - (num * 50); // 0 -> 65, 1 -> 15
      return (
        <SvgBox>
          <rect x="60" y="25" width="25" height="40" stroke="var(--halo-cyan)" fill={num < 0.85 ? "rgba(6,182,212,0.15)" : "none"} strokeWidth="2" rx="3" />
          <text x="72.5" y="45" fill={num < 0.85 ? "#fff" : "var(--text-muted)"} fontSize="10" textAnchor="middle" fontWeight="bold">85%</text>
          
          <rect x="120" y="55" width="20" height="10" stroke="#f87171" fill={num < 0.20 ? "rgba(248,113,113,0.1)" : "none"} strokeWidth="1" strokeDasharray="3 3" rx="3" />
          <text x="130" y="62" fill={num < 0.20 ? "#f87171" : "var(--text-muted)"} fontSize="9" textAnchor="middle">20%</text>
          
          <line x1="30" y1={thresholdY} x2="210" y2={thresholdY} stroke="#fff" strokeWidth="1.5" strokeDasharray="4 2" />
          <text x="180" y={thresholdY - 5} fill="#fff" fontSize="9">Threshold: {Math.round(num * 100)}%</text>
        </SvgBox>
      )
    }
    case 'sway_boxmot_match_thresh': {
      const radius = num * 40; // 0.1 to 1.0 -> 4 to 40
      return (
        <SvgBox>
          <circle cx="120" cy="40" r="4" fill="var(--halo-cyan)" />
          <circle cx="120" cy="40" r={radius} fill="rgba(6,182,212,0.1)" stroke="var(--halo-cyan)" strokeWidth="1.5" strokeDasharray="3 3" />
          <text x="120" y="75" fill="var(--text-muted)" fontSize="9" textAnchor="middle">Match acceptance radius scaling</text>
        </SvgBox>
      )
    }
    case 'sway_boxmot_max_age': {
      const frames = num;
      const width = Math.min(140, Math.max(20, frames * 1.5));
      return (
        <SvgBox>
          <rect x="25" y="20" width="20" height="40" fill="var(--halo-cyan)" opacity="0.8" />
          <rect x={45 + width} y="20" width="20" height="40" fill="var(--halo-cyan)" opacity="0.8" />
          
          <rect x="45" y="10" width={width} height="60" fill="#333" stroke="var(--text-muted)" strokeWidth="1" />
          <text x={45 + width/2} y="42" fill="var(--text-muted)" fontSize="9" textAnchor="middle">Gap ({frames}f)</text>
          
          <path d={`M 35 40 L ${55 + width} 40`} stroke="var(--halo-cyan)" strokeWidth="2" strokeDasharray="6 4" markerEnd="url(#arrow)" />
        </SvgBox>
      )
    }
    case 'spatial_outlier_std_factor': {
      const radius = num * 12; // 1 to 5 -> 12 to 60
      return (
        <SvgBox>
          <circle cx="120" cy="40" r={radius} fill="rgba(6,182,212,0.1)" stroke="var(--halo-cyan)" strokeWidth="1" strokeDasharray="4 4" />
          <circle cx="120" cy="40" r="4" fill="#fff" />
          <circle cx="110" cy="30" r="4" fill="#fff" />
          <circle cx="130" cy="45" r="4" fill="#fff" />
          
          <circle cx="180" cy="20" r="4" fill={180 - 120 > radius ? "#f87171" : "#fff"} />
          <text x="120" y="75" fill="var(--halo-cyan)" fontSize="9" textAnchor="middle">{num.toFixed(1)}σ Safe Zone</text>
        </SvgBox>
      )
    }
    case 'bbox_size_min_frac': {
      const minHeight = 40 * num;
      return (
        <SvgBox>
          <rect x="80" y="20" width="20" height="40" fill="var(--halo-cyan)" opacity="0.5" />
          <text x="90" y="75" fill="var(--text-muted)" fontSize="9" textAnchor="middle">Median Size</text>
          
          <rect x="140" y={60 - minHeight} width="20" height={minHeight} fill="none" stroke="#f87171" strokeWidth="2" strokeDasharray="2 2" />
          <text x="150" y="75" fill="#f87171" fontSize="9" textAnchor="middle">Drop if &lt; {Math.round(num * 100)}%</text>
        </SvgBox>
      )
    }
    case 'bbox_size_max_frac': {
      const maxHeight = 20 * num;
      return (
        <SvgBox>
          <rect x="80" y="30" width="20" height="40" fill="var(--halo-cyan)" opacity="0.5" />
          <text x="90" y="85" fill="var(--text-muted)" fontSize="9" textAnchor="middle">Median Size</text>
          
          <rect x="140" y={80 - maxHeight} width={10 * num} height={maxHeight} fill="none" stroke="#f87171" strokeWidth="2" strokeDasharray="2 2" />
          <text x="150" y="85" fill="#f87171" fontSize="9" textAnchor="middle">Drop if &gt; {num.toFixed(2)}x limit</text>
        </SvgBox>
      )
    }
    case 'short_track_min_frac': {
      const width = num * 160;
      return (
        <SvgBox>
          <line x1="40" y1="20" x2="200" y2="20" stroke="var(--text-muted)" strokeWidth="4" strokeLinecap="round" />
          <text x="120" y="15" fill="#fff" fontSize="9" textAnchor="middle">Full Video Track</text>
          
          <rect x="40" y="45" width={width} height="10" fill="none" stroke="#f87171" strokeWidth="2" strokeDasharray="2 2" />
          <text x={40 + width/2} y="68" fill="#f87171" fontSize="9" textAnchor="middle">Must survive {Math.round(num * 100)}% to keep</text>
        </SvgBox>
      )
    }
    case 'audience_region_x_min_frac': {
      const startX = 40 + (num * 160);
      return (
        <SvgBox>
          <rect x="40" y="10" width="160" height="60" fill="none" stroke="var(--text-muted)" strokeWidth="2" />
          <rect x={startX} y="10" width={200 - startX} height="60" fill="rgba(248,113,113,0.3)" stroke="#f87171" strokeWidth="1" />
          <text x="120" y="30" fill="var(--text-muted)" fontSize="9" textAnchor="middle">Video Frame</text>
          <text x={startX + (200 - startX)/2} y="45" fill="#f87171" fontSize="9" textAnchor="middle">X &gt; {Math.round(num * 100)}%</text>
        </SvgBox>
      )
    }
    case 'audience_region_y_min_frac': {
      const startY = 10 + (num * 60);
      return (
        <SvgBox>
          <rect x="80" y="10" width="80" height="60" fill="none" stroke="var(--text-muted)" strokeWidth="2" />
          <rect x="80" y={startY} width="80" height={70 - startY} fill="rgba(248,113,113,0.3)" stroke="#f87171" strokeWidth="1" />
          <text x="120" y="30" fill="var(--text-muted)" fontSize="9" textAnchor="middle">Video Frame</text>
          <text x="120" y={startY + (70 - startY)/2 + 4} fill="#f87171" fontSize="9" textAnchor="middle">Y &gt; {Math.round(num * 100)}%</text>
        </SvgBox>
      )
    }
    case 'pose_visibility_threshold': {
      const thresholdY = 60 - (num * 40);
      return (
        <SvgBox>
          <circle cx="100" cy="20" r="5" fill={num < 0.85 ? "var(--halo-cyan)" : "none"} stroke="var(--halo-cyan)" strokeWidth="2" />
          <text x="115" y="23" fill={num < 0.85 ? "#fff" : "var(--text-muted)"} fontSize="9">85% Keypoint</text>
          
          <circle cx="100" cy="45" r="5" fill={num < 0.20 ? "var(--halo-cyan)" : "none"} stroke="#f87171" strokeWidth="2" strokeDasharray="2 2" />
          <text x="115" y="48" fill={num < 0.20 ? "#fff" : "var(--text-muted)"} fontSize="9">20% Keypoint</text>
          
          <line x1="40" y1={thresholdY} x2="200" y2={thresholdY} stroke="#fff" strokeWidth="1.5" strokeDasharray="4 2" />
        </SvgBox>
      )
    }
    case 'dedup_min_pair_oks': {
      const circleDist = 40 * (1.0 - num);
      return (
        <SvgBox>
          <circle cx={120 - circleDist} cy="40" r="15" fill="none" stroke="var(--halo-cyan)" strokeWidth="2" />
          <circle cx={120 + circleDist} cy="40" r="15" fill="none" stroke="#f87171" strokeWidth="2" />
          <text x="120" y="70" fill="var(--text-muted)" fontSize="9" textAnchor="middle">OKS Similarity &gt; {Math.round(num * 100)}% triggers merge</text>
        </SvgBox>
      )
    }
    case 'dedup_antipartner_min_iou': {
      const overlapX = 80 + (num * 60); 
      return (
        <SvgBox>
          <rect x="60" y="20" width="40" height="40" fill="rgba(6,182,212,0.2)" stroke="var(--halo-cyan)" strokeWidth="2" />
          <rect x={overlapX} y="20" width="40" height="40" fill="rgba(248,113,113,0.2)" stroke="#f87171" strokeWidth="2" />
          <text x="120" y="75" fill="var(--text-muted)" fontSize="9" textAnchor="middle">Protects partners under {Math.round(num * 100)}% overlap</text>
        </SvgBox>
      )
    }

    // Generic fallbacks for string enums
    case 'tracker_technology':
      return (
        <SvgBox>
          <path d="M 40 40 L 90 40" stroke="var(--halo-cyan)" strokeWidth="3" markerEnd="url(#arrow)" />
          <rect x="25" y="25" width="20" height="30" stroke="var(--halo-cyan)" fill="none" strokeWidth="2" />
          <rect x="95" y="25" width="20" height="30" stroke="var(--halo-cyan)" fill="none" strokeWidth="2" />
          <text x="160" y="43" fill="#fff" fontSize="11">{String(value || 'deep_ocsort')}</text>
        </SvgBox>
      )
    default:
      return null
  }
}
