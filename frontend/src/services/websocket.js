/**
 * FedAgri WebSocket Real-Time Telemetry Hook & Manager
 */

class WebSocketManager {
  constructor() {
    this.ws = null;
    this.subscribers = new Set();
    this.statusListeners = new Set();
    this.isConnected = false;
    this.reconnectTimeout = null;
  }

  connect() {
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.port === '5173' ? 'localhost:8000' : window.location.host;
    const url = `${protocol}//${host}/ws/training`;

    try {
      this.ws = new WebSocket(url);

      this.ws.onopen = () => {
        this.isConnected = true;
        this.notifyStatus(true);
      };

      this.ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          this.subscribers.forEach((cb) => {
            try { cb(payload); } catch (e) { console.error('WS subscriber error:', e); }
          });
        } catch (e) {
          console.error('WS JSON parse error:', e);
        }
      };

      this.ws.onclose = () => {
        this.isConnected = false;
        this.notifyStatus(false);
        this.scheduleReconnect();
      };

      this.ws.onerror = () => {
        this.isConnected = false;
        this.notifyStatus(false);
      };
    } catch (e) {
      this.scheduleReconnect();
    }
  }

  scheduleReconnect() {
    if (this.reconnectTimeout) clearTimeout(this.reconnectTimeout);
    this.reconnectTimeout = setTimeout(() => this.connect(), 2500);
  }

  subscribe(callback) {
    this.subscribers.add(callback);
    return () => this.subscribers.delete(callback);
  }

  subscribeStatus(listener) {
    this.statusListeners.add(listener);
    listener(this.isConnected);
    return () => this.statusListeners.delete(listener);
  }

  notifyStatus(connected) {
    this.statusListeners.forEach((fn) => fn(connected));
  }
}

export const wsManager = new WebSocketManager();
