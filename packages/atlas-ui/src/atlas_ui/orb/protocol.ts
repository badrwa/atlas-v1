// AUTO-GENERATED from atlas_ui.protocol (v1) — do not edit by hand.
// Regenerate with: scripts/gen_ui_protocol.py (or: atlas ui protocol)

export interface HelloMessage {
  v: number;
  kind: string;
  version: string;
  states: string[];
  powers: Record<string, boolean>;
}

export interface StateMessage {
  v: number;
  kind: string;
  state: string;
  previous: string;
  mic_muted: boolean;
}

export interface CaptionMessage {
  v: number;
  kind: string;
  role: string;
  text: string;
  language: string;
  final: boolean;
}

export interface MoodMessage {
  v: number;
  kind: string;
  mood: string;
  hue: number;
  energy: number;
  warmth: number;
  humor_allowed: boolean;
}

export interface LevelMessage {
  v: number;
  kind: string;
  value: number;
}

export interface ToolMessage {
  v: number;
  kind: string;
  skill: string;
  phase: string;
  ok: boolean;
  spoken: string;
}

export interface ConfirmMessage {
  v: number;
  kind: string;
  question: string;
  timeout_s: number;
}

export interface DegradedMessage {
  v: number;
  kind: string;
  degraded: boolean;
  reason: string;
  mode: string;
}

export type AtlasMessage = HelloMessage | StateMessage | CaptionMessage | MoodMessage | LevelMessage | ToolMessage | ConfirmMessage | DegradedMessage;
