import type { ComponentType, CSSProperties } from 'react'
import {
  CheckCircle2,
  CircleDashed,
  Clock,
  Loader2,
  StopCircle,
  Unplug,
  XCircle,
} from 'lucide-react'
import type { RunInfo } from '../types'

export function statusPresentation(status: string): {
  title: string
  subtitle: string
  color: string
  Icon: ComponentType<{ size?: number; className?: string; style?: CSSProperties }>
} {
  switch (status) {
    case 'done':
      return {
        title: 'Complete',
        subtitle: 'Outputs are ready — open Watch for phase clips and render styles.',
        color: '#10b981',
        Icon: CheckCircle2,
      }
    case 'running':
      return {
        title: 'Running',
        subtitle: 'Pipeline is executing (detection → pose → export). This can take a while.',
        color: '#0ea5e9',
        Icon: Loader2,
      }
    case 'queued':
      return {
        title: 'Queued',
        subtitle: 'Waiting to start — runs start one after another when workers are free.',
        color: '#a78bfa',
        Icon: Clock,
      }
    case 'error':
      return {
        title: 'Failed',
        subtitle: 'This run stopped with an error. Check the message below or server logs.',
        color: '#ef4444',
        Icon: XCircle,
      }
    case 'cancelled':
      return {
        title: 'Stopped',
        subtitle: 'You stopped this run from the Lab. Partial outputs may exist under the run folder.',
        color: '#fbbf24',
        Icon: StopCircle,
      }
    default:
      return {
        title: status || 'Unknown',
        subtitle: 'Status not reported yet — try refreshing in a few seconds.',
        color: '#94a3b8',
        Icon: CircleDashed,
      }
  }
}

export function statusPresentationForRun(
  run: Pick<RunInfo, 'status' | 'subprocess_alive'>,
): ReturnType<typeof statusPresentation> {
  if (run.status === 'running' && run.subprocess_alive === false) {
    return {
      title: 'Stale (not attached)',
      subtitle:
        'The run folder still looks in-progress, but this API process is not running main.py for it (common after restarting uvicorn). Delete the run or start a new batch — Stop is unavailable.',
      color: '#f59e0b',
      Icon: Unplug,
    }
  }
  return statusPresentation(run.status)
}
