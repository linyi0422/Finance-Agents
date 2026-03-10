# GitHub / VS Code 403 排查记录

适用场景：
- 浏览器能访问 GitHub，但 VS Code、`gh auth login`、`git push` 失败
- 报错表现为 `403`、超时、设备码登录后换 token 失败

## 这次排查的结论

本机代理程序正常运行，但 `VS Code`、`git`、`gh` 没有接入代理。

已确认的事实：
- 本机运行了 `FlClash`
- 本地代理端口 `127.0.0.1:7890` 正在监听
- 通过代理访问 `https://github.com` 返回 `200`
- 之前这些位置都没有代理配置：
  - Windows 系统代理
  - WinHTTP
  - VS Code `http.proxy`
  - `git config --global http.proxy`
  - `HTTP_PROXY` / `HTTPS_PROXY`

核心判断：
- 浏览器可能走了插件代理
- 但 VS Code、Git、GitHub CLI 不会自动继承浏览器插件代理
- 所以它们仍在直连 GitHub，导致 `403` 或超时

## 已验证有效的配置

### 1. Git 全局代理

```powershell
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890
```

### 2. VS Code 用户设置

文件：
- `%APPDATA%\\Code\\User\\settings.json`

加入：

```json
{
  "http.proxy": "http://127.0.0.1:7890",
  "http.proxySupport": "override",
  "http.systemCertificates": true
}
```

如果后续遇到证书错误，再临时补：

```json
"http.proxyStrictSSL": false
```

说明：
- 只有遇到 SSL/证书报错时才建议加这一项
- 默认不要先关严格校验

### 3. 用户级环境变量

```powershell
[Environment]::SetEnvironmentVariable('HTTP_PROXY','http://127.0.0.1:7890','User')
[Environment]::SetEnvironmentVariable('HTTPS_PROXY','http://127.0.0.1:7890','User')
[Environment]::SetEnvironmentVariable('NO_PROXY','localhost,127.0.0.1','User')
```

说明：
- 这一步主要给 `gh`、部分扩展和新开的终端使用
- 配完后最好重开 VS Code 或新开终端

## 快速验证方法

### 验证 GitHub 是否能通过代理访问

```powershell
Invoke-WebRequest -Uri "https://github.com" -Proxy "http://127.0.0.1:7890" -UseBasicParsing
```

成功标准：
- 返回状态码 `200`

### 验证本地代理端口是否在监听

```powershell
netstat -ano | findstr LISTENING | findstr ":7890"
```

### 验证 Git 代理配置

```powershell
git config --global --get http.proxy
git config --global --get https.proxy
```

## 后续登录 GitHub 的建议流程

1. 确认 `FlClash` 正在运行
2. 确认 `127.0.0.1:7890` 可用
3. 重开 VS Code 或新开终端
4. 执行：

```powershell
gh auth login --hostname github.com --git-protocol https --web --skip-ssh-key
```

如果网页授权成功但 CLI 仍失败，优先检查：
- 当前终端是否继承了新的 `HTTP_PROXY` / `HTTPS_PROXY`
- VS Code 是否已重启
- 代理软件是否切换了端口

## 当前遗留问题

网络链路已经打通，但仓库提交前还需要 Git 身份配置：

```powershell
git config user.name "YOUR_NAME"
git config user.email "YOUR_EMAIL"
```

没有这两项时，`git commit` 会失败，但这不是网络问题。
