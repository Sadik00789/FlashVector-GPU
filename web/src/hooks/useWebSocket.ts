'use client';

import { useEffect, useRef, useState, useCallback } from 'react';
import { QueryParams, SearchResponse } from '../lib/types';

function resolveWsUrl(): string {
  const explicit = process.env.NEXT_PUBLIC_WS_URL;
  if (explicit) return explicit;

  const apiUrl = process.env.NEXT_PUBLIC_API_URL;
  if (apiUrl) {
    try {
      const url = new URL(apiUrl);
      url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
      url.pathname = '/ws/stream';
      url.search = '';
      url.hash = '';
      return url.toString();
    } catch {
      // Fall through to host/port resolution
    }
  }

  const host = typeof window !== 'undefined' ? window.location.hostname : 'localhost';
  const protocol =
    typeof window !== 'undefined' && window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const port = process.env.NEXT_PUBLIC_WS_PORT || '8080';
  return `${protocol}//${host}:${port}/ws/stream`;
}

export function useWebSocket() {
  const [isConnected, setIsConnected] = useState(false);
  const [latestResponse, setLatestResponse] = useState<SearchResponse | null>(null);
  const [latencyHistory, setLatencyHistory] = useState<number[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const retryDelayRef = useRef<number>(1000);
  const manuallyClosedRef = useRef(false);

  const connect = useCallback(() => {
    try {
      const ws = new WebSocket(resolveWsUrl());

      ws.onopen = () => {
        setIsConnected(true);
        retryDelayRef.current = 1000; // Reset backoff delay on successful connect
      };

      ws.onmessage = (event) => {
        try {
          const data: SearchResponse = JSON.parse(event.data);
          setLatestResponse(data);
          setLatencyHistory((prev) => {
            const next = [...prev, data.latency_us];
            if (next.length > 40) next.shift();
            return next;
          });
        } catch {
          // ignore malformed message
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        wsRef.current = null;

        if (manuallyClosedRef.current) return;

        // Exponential backoff with jitter up to 10s
        const nextDelay = Math.min(10000, retryDelayRef.current * 2);
        retryDelayRef.current = nextDelay;
        reconnectTimeoutRef.current = setTimeout(connect, nextDelay);
      };

      ws.onerror = () => {
        ws.close();
      };

      wsRef.current = ws;
    } catch {
      const nextDelay = Math.min(10000, retryDelayRef.current * 2);
      retryDelayRef.current = nextDelay;
      reconnectTimeoutRef.current = setTimeout(connect, nextDelay);
    }
  }, []);

  useEffect(() => {
    manuallyClosedRef.current = false;
    connect();
    return () => {
      manuallyClosedRef.current = true;
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [connect]);

  const sendQuery = useCallback((params: QueryParams) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(params));
    }
  }, []);

  return {
    isConnected,
    latestResponse,
    latencyHistory,
    sendQuery,
  };
}
