const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://invensight.onrender.com';

export interface UploadResponse {
  message: string;
  user_id: string;
  filename: string;
}

export interface QueryResponse {
  user_id: string;
  query: string;
  answer: string;
}

export interface FileInfo {
  file_name: string;
  s3_key: string;
  size: number;
  last_modified: string;
}

export const api = {
  // Upload PDF
  async uploadPDF(userId: string, file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('user_id', userId);
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/upload/`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Upload failed');
    }

    return response.json();
  },

  // Query
  async query(userId: string, query: string): Promise<QueryResponse> {
    const response = await fetch(`${API_BASE_URL}/query/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        user_id: userId,
        query: query,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Query failed');
    }

    return response.json();
  },

  // List user files
  async listFiles(userId: string): Promise<FileInfo[]> {
    const response = await fetch(`${API_BASE_URL}/users/${userId}/files`, {
      method: 'GET',
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to list files');
    }

    return response.json();
  },

  // Delete file
  async deleteFile(userId: string, fileName: string): Promise<{ message: string }> {
    const response = await fetch(`${API_BASE_URL}/users/${userId}/files/${fileName}`, {
      method: 'DELETE',
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to delete file');
    }

    return response.json();
  },

  // Health check
  async healthCheck(): Promise<{ status: string; message: string }> {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.json();
  },

  // Check upload status
  async checkUploadStatus(userId: string, filename: string): Promise<{ status: string }> {
    const response = await fetch(
      `${API_BASE_URL}/upload/status/${userId}/${filename}`
    );
    return response.json();
  },
};