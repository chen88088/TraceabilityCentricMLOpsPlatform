# 使用 Node.js 18 作為基底映像
FROM node:18

# 設定工作目錄
WORKDIR /app

# 複製 package.json 和 package-lock.json
COPY package*.json ./

# 安裝依賴套件
RUN npm install

# 複製其他程式碼
COPY . .

# 開發模式：啟動 Vite
EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host"]
