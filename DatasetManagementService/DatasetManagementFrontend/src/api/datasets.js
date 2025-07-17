import axios from "axios";

// 使用 /api 作為前綴，讓 Vite 的 proxy 處理轉發
const API_BASE_URL = "/api/datasets";

// const API_BASE_URL = "http://192.168.158.43:8082/datasets";

// 分類相關 API
export const fetchCategories = () => axios.get(`${API_BASE_URL}/categories`);
export const fetchCategory = (categoryId) => axios.get(`${API_BASE_URL}/categories/${categoryId}`);
export const createCategory = (data) => axios.post(`${API_BASE_URL}/categories`, data, { headers: { "Cache-Control": "no-cache" }});
export const deleteCategory = (categoryId) => axios.delete(`${API_BASE_URL}/categories/${categoryId}`);

// 版本相關 API
export const fetchVersions = (categoryName) => axios.get(`${API_BASE_URL}/${categoryName}/versions`);
export const fetchVersion = (categoryName, version) => axios.get(`${API_BASE_URL}/${categoryName}/versions/${version}`);
export const createVersion = (categoryName, data) => axios.post(`${API_BASE_URL}/${categoryName}/versions`, data);
export const deleteVersion = (categoryName, version) => axios.delete(`${API_BASE_URL}/${categoryName}/versions/${version}`);
