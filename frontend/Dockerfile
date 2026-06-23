FROM node:20-alpine AS builder
LABEL stage=builder

WORKDIR /app
COPY package.json .
RUN npm install
COPY . .

ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

# ── Production image: Nginx serving the built assets ──────────────
FROM nginx:1.27-alpine
LABEL maintainer="CodeMigrateAI — MTech Final Year Project"

COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]