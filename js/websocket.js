/**
 * FedAgri WebSocket Real-Time Telemetry Client
 */

class FedAgriWebSocket {
  constructor() {
    this.ws = null;
    this.handlers = new Map();
    this.statusListeners = [];
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 30;
    this.reconnectInterval = 2500;
    this.isConnected = false;
  }

  connect() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/training`;

    console.log(`[WebSocket] Connecting to ${wsUrl}...`);
    try {
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        console.log("[WebSocket] Connected successfully.");
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this.notifyStatus(true);
      };

      this.ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          this.dispatchEvent(payload.type, payload);
          // Also dispatch wildcard
          this.dispatchEvent("*", payload);
        } catch (e) {
          console.error("[WebSocket] Message parsing error:", e);
        }
      };

      this.ws.onclose = () => {
        console.warn("[WebSocket] Connection closed. Attempting reconnect...");
        this.isConnected = false;
        this.notifyStatus(false);
        this.scheduleReconnect();
      };

      this.ws.onerror = (err) => {
        console.error("[WebSocket] Socket error:", err);
        this.isConnected = false;
        this.notifyStatus(false);
      };
    } catch (err) {
      console.error("[WebSocket] Exception opening socket:", err);
      this.scheduleReconnect();
    }
  }

  scheduleReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      setTimeout(() => this.connect(), this.reconnectInterval);
    }
  }

  on(eventType, handler) {
    if (!this.handlers.has(eventType)) {
      this.handlers.set(eventType, []);
    }
    this.handlers.get(eventType).push(handler);
  }

  off(eventType, handler) {
    if (!this.handlers.has(eventType)) return;
    const list = this.handlers.get(eventType);
    const idx = list.indexOf(handler);
    if (idx !== -1) list.splice(idx, 1);
  }

  dispatchEvent(eventType, data) {
    if (this.handlers.has(eventType)) {
      this.handlers.get(eventType).forEach(fn => {
        try {
          fn(data);
        } catch (e) {
          console.error(`Error in WebSocket handler for ${eventType}:`, e);
        }
      });
    }
  }

  onStatusChange(fn) {
    this.statusListeners.push(fn);
  }

  notifyStatus(connected) {
    this.statusListeners.forEach(fn => fn(connected));
  }
}

window.fedWs = new FedAgriWebSocket();
