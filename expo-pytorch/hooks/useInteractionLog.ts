/**
 * useInteractionLog.ts
 *
 * Logs user interactions with feed items to a JSONL file on device.
 * Each line is a JSON object: { contentId, timestamp, viewDurationMs, action, reward, ... }
 *
 * This is the data collection layer for the offline RL pipeline.
 * Export the log file and feed it to scripts/offline_rl_pipeline.py.
 *
 * Reward design:
 *   report             → -5.0  (strong negative — user found content harmful)
 *   skip (< 2s view)   → -0.1  (mild negative — content wasn't engaging)
 *   view (2–5s)        → +0.3  (mild positive)
 *   view (> 5s)        → +1.0  (strong positive — high engagement)
 */

import { File, Paths } from 'expo-file-system';
import { useCallback, useRef } from 'react';
import type { ReportCategory } from '../data/feedData';

export type InteractionAction = 'view' | 'skip' | 'report';

export type Interaction = {
  contentId: string;
  contentUri: string;
  timestamp: number;        // Unix ms
  viewDurationMs: number;
  action: InteractionAction;
  reportCategory?: ReportCategory;
  reward: number;           // scalar reward for offline RL
};

const getLogFile = () => new File(Paths.document, 'interaction_log.jsonl');

function computeReward(
  action: InteractionAction,
  viewDurationMs: number,
): number {
  if (action === 'report') return -5.0;
  const secs = viewDurationMs / 1000;
  if (secs < 2) return -0.1;
  if (secs < 5) return 0.3;
  return 1.0;
}

export function useInteractionLog() {
  const viewStartRef = useRef<number>(Date.now());

  /** Call when a new item becomes visible in the feed. */
  const startView = useCallback(() => {
    viewStartRef.current = Date.now();
  }, []);

  /**
   * Append one interaction to the on-device log.
   * viewDurationMs is computed from the last startView() call unless overridden.
   */
  const logInteraction = useCallback(async (
    contentId: string,
    contentUri: string,
    action: InteractionAction,
    reportCategory?: ReportCategory,
  ): Promise<void> => {
    const viewDurationMs = Date.now() - viewStartRef.current;
    const reward = computeReward(action, viewDurationMs);

    const entry: Interaction = {
      contentId,
      contentUri,
      timestamp: Date.now(),
      viewDurationMs,
      action,
      ...(reportCategory ? { reportCategory } : {}),
      reward,
    };

    try {
      const line = JSON.stringify(entry) + '\n';
      const file = getLogFile();
      const existing = file.exists ? await file.text() : '';
      file.write(existing + line);
    } catch (e) {
      console.warn('[InteractionLog] write failed:', e);
    }
  }, []);

  /** Returns the on-device path to the JSONL log file for sharing/export. */
  const getLogPath = useCallback((): string => getLogFile().uri, []);

  /** Reads and parses all logged interactions. */
  const getLogs = useCallback(async (): Promise<Interaction[]> => {
    try {
      const file = getLogFile();
      if (!file.exists) return [];
      const content = await file.text();
      return content
        .trim()
        .split('\n')
        .filter(Boolean)
        .map((line) => JSON.parse(line) as Interaction);
    } catch {
      return [];
    }
  }, []);

  /** Clears the log file. */
  const clearLogs = useCallback(async (): Promise<void> => {
    try { getLogFile().write(''); } catch { /* ignore if file doesn't exist */ }
  }, []);

  return { startView, logInteraction, getLogPath, getLogs, clearLogs };
}
