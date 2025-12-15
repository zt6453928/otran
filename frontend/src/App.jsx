import { useEffect, useRef, useState } from 'react'
import { Menu, Paperclip, Link as LinkIcon, Info, FileText, FileImage, FileType, Loader2, AlertTriangle } from 'lucide-react'

const TerminalLogo = () => {
  const sequences = [
    { char: '<', color: '#2563eb' },
    { char: 'A', color: '#0f172a' },
    { char: 'i', color: '#0f172a' },
    { char: 'l', color: '#0f172a' },
    { char: 'a', color: '#0f172a' },
    { char: 't', color: '#0f172a' },
    { char: ' ', color: '#0f172a' },
    { char: '/', color: '#94a3b8' },
    { char: '>', color: '#2563eb' }
  ]
  const [displayedChars, setDisplayedChars] = useState([])
  const [isTyping, setIsTyping] = useState(true)

  useEffect(() => {
    let timeoutId
    let currentIndex = 0
    let isDeleting = false

    const typeLoop = () => {
      if (!isDeleting) {
        if (currentIndex < sequences.length) {
          setIsTyping(true)
          setDisplayedChars(sequences.slice(0, currentIndex + 1))
          currentIndex += 1
          timeoutId = setTimeout(typeLoop, Math.random() * 100 + 50)
        } else {
          setIsTyping(false)
          isDeleting = true
          timeoutId = setTimeout(typeLoop, 3000)
        }
      } else {
        setDisplayedChars([])
        currentIndex = 0
        isDeleting = false
        timeoutId = setTimeout(typeLoop, 500)
      }
    }

    timeoutId = setTimeout(typeLoop, 500)
    return () => clearTimeout(timeoutId)
  }, [])

  return (
    <div className="flex items-center relative">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@600&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@900&display=swap');
        @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }
        @keyframes flow-gradient {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
        .animate-flow { background-size: 200% auto; animation: flow-gradient 3s linear infinite; }
      `}</style>
      <div className="font-['Fira_Code'] text-xl font-semibold leading-none flex items-center select-none tracking-tight">
        {displayedChars.map((item, index) => (
          <span key={index} style={{ color: item.color }}>{item.char}</span>
        ))}
        <span
          className="inline-block w-2.5 h-6 ml-0.5 bg-slate-800 align-middle"
          style={{ animation: isTyping ? 'none' : 'blink 1s step-end infinite', opacity: isTyping ? 1 : undefined }}
        ></span>
      </div>
    </div>
  )
}

function App() {
  const [isHovering, setIsHovering] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const [fileName, setFileName] = useState('')
  const fileInputRef = useRef(null)

  const handleLocalClick = () => fileInputRef.current?.click()

  const handleFileChange = (event) => {
    const file = event.target.files?.[0]
    if (file) {
      uploadFile(file)
    }
    event.target.value = ''
  }

  const handleDrop = (event) => {
    event.preventDefault()
    setIsHovering(false)
    const file = event.dataTransfer.files?.[0]
    if (file) {
      uploadFile(file)
    }
  }

  const uploadFile = async (file) => {
    setError('')
    setMessage('正在上传文件...')
    setUploading(true)
    setFileName(file.name)
    try {
      const formData = new FormData()
      formData.append('file', file)
      const response = await fetch('/api/parse', {
        method: 'POST',
        body: formData
      })
      const data = await response.json()
      if (!response.ok) {
        throw new Error(data.error || '上传失败')
      }
      if (data.task_id) {
        setMessage('上传成功，正在跳转...')
        const targetUrl = new URL(window.location.href)
        if (targetUrl.port === '5173') {
          targetUrl.port = '8080'
        }
        targetUrl.pathname = '/viewer'
        targetUrl.search = `?task_id=${data.task_id}`
        window.location.href = targetUrl.toString()
      } else {
        throw new Error('未能获取任务ID')
      }
    } catch (err) {
      setError(err.message)
      setMessage('')
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="min-h-screen bg-white font-sans text-slate-900 selection:bg-blue-100 relative overflow-hidden">
      <div className="absolute inset-0 z-0 h-full w-full bg-white bg-[radial-gradient(#e5e7eb_1px,transparent_1px)] [background-size:20px_20px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)] pointer-events-none"></div>
      <header className="relative z-10 flex items-center justify-between px-6 py-4 md:px-8">
        <div className="flex items-center gap-4">
          <button className="p-2 hover:bg-slate-100 rounded-lg transition-colors">
            <Menu className="w-6 h-6 text-slate-700" />
          </button>
          <div className="flex items-center gap-3 cursor-pointer">
            <div className="w-10 h-10 bg-black rounded-lg flex items-center justify-center relative overflow-hidden shrink-0">
              <svg viewBox="0 0 100 100" className="w-7 h-7 text-white" fill="currentColor" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round">
                <path d="M 20 75 C 20 75, 40 85, 55 65 C 70 45, 45 35, 45 35 C 45 35, 75 25, 85 55" strokeWidth="12" fill="none" />
                <circle cx="85" cy="25" r="7" stroke="none" />
              </svg>
            </div>
            <TerminalLogo />
          </div>
        </div>
      </header>

      <main className="relative z-10 flex flex-col items-center justify-center pt-12 px-4 md:pt-20">
        <div className="text-center mb-10 space-y-3">
          <h1 className="text-4xl md:text-5xl font-bold text-slate-900 tracking-tight">文档解析</h1>
          <p className="text-slate-500 text-lg md:text-xl font-light">全格式兼容 · 精准提取 · 极速输出</p>
        </div>

        <div
          className={`w-full max-w-4xl bg-white/80 backdrop-blur-sm rounded-3xl border-2 border-dashed transition-all duration-300 ease-in-out flex flex-col items-center justify-center py-20 px-6 relative overflow-hidden ${isHovering ? 'border-blue-400 bg-blue-50/30' : 'border-slate-200'}`}
          onDragOver={(e) => { e.preventDefault(); setIsHovering(true) }}
          onDragLeave={() => setIsHovering(false)}
          onDrop={handleDrop}
        >
          <div className="mb-10 relative w-32 h-24 flex items-center justify-center">
            <div className="absolute transform -rotate-12 -translate-x-8 translate-y-2 bg-blue-100 p-3 rounded-lg shadow-sm border border-blue-200">
              <FileText className="w-8 h-8 text-blue-500" />
            </div>
            <div className="absolute transform rotate-12 translate-x-8 translate-y-2 bg-green-100 p-3 rounded-lg shadow-sm border border-green-200">
              <FileImage className="w-8 h-8 text-green-500" />
            </div>
            <div className="relative z-10 bg-red-50 p-4 rounded-xl shadow-lg border border-red-100 transform hover:-translate-y-1 transition-transform">
              <FileType className="w-10 h-10 text-red-500" />
              <div className="absolute -bottom-1 -right-1 bg-white rounded-full p-1 shadow-sm border border-slate-100">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
              </div>
            </div>
          </div>

          <div className="flex flex-col sm:flex-row gap-4 w-full max-w-md justify-center mb-8">
            <button onClick={handleLocalClick} className="flex items-center justify-center gap-2 px-6 py-3 bg-white border border-slate-200 rounded-xl shadow-sm hover:shadow-md hover:border-slate-300 hover:bg-slate-50 transition-all group">
              <Paperclip className="w-5 h-5 text-slate-600 group-hover:text-black" />
              <span className="text-slate-700 font-medium group-hover:text-black">本地上传</span>
            </button>
            <button className="flex items-center justify-center gap-2 px-6 py-3 bg-white border border-slate-200 rounded-xl shadow-sm hover:shadow-md hover:border-slate-300 hover:bg-slate-50 transition-all group">
              <LinkIcon className="w-5 h-5 text-slate-600 group-hover:text-black" />
              <span className="text-slate-700 font-medium group-hover:text-black">URL 上传</span>
            </button>
          </div>

          <div className="text-slate-400 text-sm flex items-center gap-1.5 cursor-default">
            <span>点击或拖拽上传</span>
            <div className="group relative">
              <Info className="w-4 h-4 cursor-help hover:text-slate-600 transition-colors" />
              <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-48 p-2 bg-slate-800 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                支持 PDF, Word, Markdown, 图片等多种格式解析。
              </div>
            </div>
          </div>

          <div className="absolute bottom-0 left-0 right-0 py-3 bg-slate-50/80 border-t border-dashed border-slate-200 flex items-center justify-start px-6">
            <div className="relative group cursor-default">
              <div className="absolute -inset-1 bg-gradient-to-r from-blue-400 via-purple-500 to-pink-500 rounded-lg blur opacity-0 group-hover:opacity-30 transition duration-500 animate-flow"></div>
              <span className="relative font-['Inter'] font-black text-sm tracking-tighter text-transparent bg-clip-text bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 animate-flow select-none">
                Ailat VLM
              </span>
            </div>
          </div>
        </div>

        {(message || error) && (
          <div className="mt-6 w-full max-w-3xl">
            <div className="bg-white/80 border border-slate-200 rounded-2xl p-4 shadow-sm flex items-center gap-3 text-sm text-slate-600">
              {uploading && <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />}
              {error && !uploading && <AlertTriangle className="w-5 h-5 text-red-500" />}
              <div>
                {message && <p className="text-slate-700">{message}</p>}
                {fileName && <p className="text-slate-400 mt-1">文件：{fileName}</p>}
                {error && <p className="text-red-500 mt-1">{error}</p>}
              </div>
            </div>
          </div>
        )}
      </main>

      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf,.doc,.docx,.ppt,.pptx,.md,.txt,.png,.jpg,.jpeg"
        className="hidden"
        onChange={handleFileChange}
      />
    </div>
  )
}

export default App
