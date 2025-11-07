---
title: "hugo搭建个人网站笔记"
---

Hugo 是一个非常强大的静态网站生成器，非常适合用来搭建个人知识库、技术博客或学术笔记站点。你提到的需求（支持图片、表格、数学公式）在 Hugo 中都可以很好地实现。

下面我们用 PaperMod 创建一个最小可用站点，包含图片、表格和数学公式。

## 1. 创建新站点

```bash
hugo new site simple_demo
cd simple_demo
```

## 2. 安装

```bash
git init
git submodule add https://github.com/adityatelange/hugo-PaperMod.git themes/PaperMod
# 如果网络下载有问题，可以添加代理下载
git submodule add https://gh-proxy.com/https://github.com/adityatelange/hugo-PaperMod.git themes/PaperMod
```

## 3. 配置 config.yaml

hugo提供了默认的`hugo.toml`文件，我偏向于使用`config.yaml`文件（记得删除`hugo.toml`文件）

```yaml
baseURL: "http://localhost:1313/" # 如果是部署到github，这里修改为https://username.github.io/
languageCode: "zh-CN"
title: "我的工作笔记"
theme: "PaperMod"

enableInlineShortcodes: true
enableEmoji: true

# 启用 KaTeX 数学公式支持
markup:
  goldmark:
    renderer:
      unsafe: true  # 允许 HTML（如 <img>）
  highlight:
    noClasses: false
  math:
    enable: true
    useKaTeX: true

params:
  env: production
  title: "我的工作笔记"
  description: "记录论文阅读与代码实践"
  author: "你的名字"
```

## 4. 创建一篇笔记

```bash
hugo new posts/reading-paper-2025.md

# 编辑 content/posts/reading-paper-2025.md：
---
title: "论文阅读：Attention Is All You Need"
date: 2025-11-06T18:00:00+08:00
draft: false
math: true
---

本文简要记录对经典论文《Attention Is All You Need》的理解。

## 数学公式示例

Transformer 中的缩放点积注意力公式：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

行内公式：$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$。

## 表格示例

| Layer | Output Size | Params |
|-------|-------------|--------|
| Embedding | 512 | 1M |
| Encoder | 512 | 60M |
| Decoder | 512 | 60M |

## 插入图片

将图片放在 `static/images/` 目录下，例如 `static/images/transformer.png`，然后引用：

![Transformer 架构](/images/transformer.png)

> 提示：确保图片文件已放入 `static/images/` 文件夹。
```

## 5. 启动本地预览

```bash
hugo server -D
```
## FAQ

1. 按照上面操作后，发现数学公式并没有被显示出来。
   
    问题分析：说明当前的PaperMod主题默认不支持数学公式，我们可以自己在合适的位置插入脚本

    1. Hugo允许自定主题模板，并覆盖当前主题的模板。

    ```bash
    mkdir -p layouts/_default/
    cp themes/PaperMod/layouts/_default/single.html layouts/_default/single.html
    ```

    2. 编辑 layouts/_default/single.html，打开这个文件，在 <head> 结束前或 <body> 开始后，添加以下代码：

        ```html
        {{- define "main" }}
        
        <article class="post-single">
        <!-- 插入的代码-开始 -->
        {{- if .Params.math }}
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
            <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
            <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
            <script>
                document.addEventListener("DOMContentLoaded", function () {
                renderMathInElement(document.body, {
                    delimiters: [
                    { left: "$$", right: "$$", display: true },
                    { left: "$", right: "$", display: false }
                    ],
                    trust: true,
                    throwOnError: false
                });
                });
            </script>
        {{- end }}
        <!-- 插入的代码-结束 -->
        <header class="post-header">
            {{ partial "breadcrumbs.html" . }}
        ...
        ```

