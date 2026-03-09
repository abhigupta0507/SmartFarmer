import apiService from './api.service';
import { API_BASE_URL } from '../constants';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { STORAGE_KEYS } from '../constants';

/**
 * Chat service — stateless (no persistence).
 * The full conversation history is sent on every request so the AI has context.
 */
class ChatService {
  /**
   * Send a text message with full conversation history for context.
   * @param {Array<{role: 'user'|'assistant', content: string}>} messages
   * @returns {Promise<string>} AI reply text
   */
  async sendMessage(messages) {
    const response = await apiService.post('/chat', { messages });
    return response.reply;
  }

  /**
   * Send a voice recording with conversation history.
   * Returns { userText, replyText, audioBuffer }
   *
   * @param {string} audioUri - Local file URI of the recorded audio
   * @param {Array<{role: 'user'|'assistant', content: string}>} history - Previous messages (excluding current)
   * @returns {Promise<{userText: string, replyText: string, audioBlob: Blob}>}
   */
  async sendVoiceMessage(audioUri, history = []) {
    const token = await AsyncStorage.getItem(STORAGE_KEYS.ACCESS_TOKEN);

    const formData = new FormData();
    formData.append('audio', {
      uri: audioUri,
      type: 'audio/m4a',
      name: 'voice-message.m4a',
    });

    // Send conversation history as JSON string in form data
    if (history.length > 0) {
      formData.append('history', JSON.stringify(history.slice(-18)));
    }

    const response = await fetch(`${API_BASE_URL}/voice-chat`, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        // Do NOT set Content-Type — let fetch set multipart/form-data boundary automatically
      },
      body: formData,
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.error || 'Voice processing failed');
    }

    // Decode text from response headers (base64 encoded to avoid encoding issues)
    const replyTextB64 = response.headers.get('X-Reply-Text');
    const userTextB64 = response.headers.get('X-User-Text');

    const replyText = replyTextB64
      ? decodeURIComponent(escape(atob(replyTextB64)))
      : '';
    const userText = userTextB64
      ? decodeURIComponent(escape(atob(userTextB64)))
      : '';

    const audioBlob = await response.blob();

    return { userText, replyText, audioBlob };
  }
}

export default new ChatService();
