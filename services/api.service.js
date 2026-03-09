import { API_BASE_URL } from "../constants";

/**
 * Lightweight API client — no auth tokens, just plain fetch.
 * Profile lives in AsyncStorage (local-only); only AI features hit the server.
 */
class ApiService {
  constructor() {
    this.baseURL = API_BASE_URL;
  }

  async _getHeaders() {
    return { "Content-Type": "application/json" };
  }

  async _request(endpoint, options = {}) {
    const headers = await this._getHeaders();
    const config = { headers, ...options };

    let response;
    try {
      response = await fetch(`${this.baseURL}${endpoint}`, config);
    } catch (networkErr) {
      throw { message: "Network error. Check your connection.", status: 0 };
    }

    let data;
    try {
      data = await response.json();
    } catch {
      throw { message: "Invalid server response.", status: response.status };
    }

    if (!response.ok) {
      throw {
        message: data?.message || "Request failed",
        errors: data?.errors ?? null,
        status: response.status,
      };
    }

    return data;
  }

  get(endpoint) {
    return this._request(endpoint, { method: "GET" });
  }

  post(endpoint, body) {
    return this._request(endpoint, {
      method: "POST",
      body: JSON.stringify(body),
    });
  }

  put(endpoint, body) {
    return this._request(endpoint, {
      method: "PUT",
      body: JSON.stringify(body),
    });
  }

  delete(endpoint) {
    return this._request(endpoint, { method: "DELETE" });
  }
}

export default new ApiService();
