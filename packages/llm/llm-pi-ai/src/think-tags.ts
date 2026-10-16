/**
 * Leading think-tag normalization for explicitly configured Chat Completions models.
 * @module dsh-llm-pi-ai/think-tags
 */

import { EMPTY_RESPONSE_CODE } from '@deepseek-ai/dsh-llm'
import type { StreamChunk } from '@deepseek-ai/dsh-llm'

const OPEN = '<think>'
const CLOSE = '</think>'

type TextKind = 'text' | 'reasoning'
type IndexedChunk = Extract<StreamChunk, { index: number }>

/** One native text block, with at most one leading reasoning section. */
class TaggedText {
  private mode: 'prefix' | TextKind = 'prefix'
  private pending = ''
  private active: { index: number; kind: TextKind; text: string } | undefined

  constructor(private readonly start: (kind: TextKind) => number) {}

  private * write(kind: TextKind, text: string): Generator<StreamChunk> {
    if (text.length === 0) return
    if (this.active === undefined) {
      this.active = { index: this.start(kind), kind, text: '' }
      yield { type: 'block-start', index: this.active.index, blockType: kind }
    }
    this.active.text += text
    yield { type: kind === 'text' ? 'text-delta' : 'reasoning-delta', index: this.active.index, text }
  }

  private * close(): Generator<StreamChunk> {
    if (this.active === undefined) return
    const { index, kind, text } = this.active
    yield { type: 'block-end', index, block: { type: kind, text } }
    this.active = undefined
  }

  * push(text: string): Generator<StreamChunk> {
    this.pending += text
    if (this.mode === 'prefix') {
      if (OPEN.startsWith(this.pending) && this.pending.length < OPEN.length) return
      if (this.pending.startsWith(OPEN)) {
        this.mode = 'reasoning'
        this.pending = this.pending.slice(OPEN.length)
      } else {
        this.mode = 'text'
      }
    }
    if (this.mode === 'reasoning') {
      const end = this.pending.indexOf(CLOSE)
      if (end < 0) {
        let held = Math.min(CLOSE.length - 1, this.pending.length)
        while (held > 0 && !CLOSE.startsWith(this.pending.slice(-held))) held--
        yield* this.write('reasoning', this.pending.slice(0, this.pending.length - held))
        this.pending = this.pending.slice(this.pending.length - held)
        return
      }
      yield* this.write('reasoning', this.pending.slice(0, end))
      yield* this.close()
      this.pending = this.pending.slice(end + CLOSE.length)
      this.mode = 'text'
    }
    if (this.pending.length > 0) yield* this.write('text', this.pending)
    this.pending = ''
  }

  * end(): Generator<StreamChunk> {
    if (this.pending.length > 0) yield* this.write(this.mode === 'reasoning' ? 'reasoning' : 'text', this.pending)
    this.pending = ''
    yield* this.close()
  }
}

/**
 * Split a leading exact `<think>` section into reasoning and ordinary text.
 * Tags elsewhere in text remain literal. An unclosed section remains reasoning
 * through block termination, including a tool call, cancellation, or failure.
 * Native reasoning and tool calls retain their metadata; split blocks receive
 * aligned Chat Completions replay metadata without changing response identity.
 * @param chunks - one Chat Completions response translated by toStreamChunks.
 * A text block's first nonempty output precedes interleaved blocks buffered
 * during tag detection; later split blocks follow their actual arrival order.
 * @returns normalized chunks with usage preserved; an otherwise successful
 *   response with no retained blocks finishes with EMPTY_RESPONSE.
 */
export async function* parseThinkTagStream(chunks: AsyncIterable<StreamChunk>): AsyncGenerator<StreamChunk> {
  const texts = new Map<number, TaggedText>()
  const indexes = new Map<number, number>()
  const blocks: { source: number; kind?: TextKind }[] = []
  const delayed: IndexedChunk[] = []
  let waitingForText: number | undefined
  const start = (source: number, kind?: TextKind): number => {
    const index = blocks.length
    blocks.push({ source, ...kind === undefined ? {} : { kind } })
    return index
  }
  function* accept(chunk: IndexedChunk): Generator<StreamChunk> {
    if (waitingForText !== undefined && chunk.index !== waitingForText) {
      delayed.push(chunk)
      return
    }
    if (chunk.type === 'block-start') {
      if (chunk.blockType === 'text') {
        waitingForText = chunk.index
        texts.set(chunk.index, new TaggedText((kind) => {
          waitingForText = undefined
          return start(chunk.index, kind)
        }))
      } else {
        const index = start(chunk.index)
        indexes.set(chunk.index, index)
        yield { ...chunk, index }
      }
    } else if (chunk.type === 'text-delta') {
      yield* (texts.get(chunk.index) as TaggedText).push(chunk.text)
    } else if (chunk.type === 'block-end' && chunk.block.type === 'text') {
      yield* (texts.get(chunk.index) as TaggedText).end()
      texts.delete(chunk.index)
      waitingForText = undefined
    } else {
      yield { ...chunk, index: indexes.get(chunk.index) as number }
    }
    if (waitingForText === undefined) {
      for (const queued of delayed.splice(0)) yield* accept(queued)
    }
  }
  function* settle(): Generator<StreamChunk> {
    while (texts.size > 0 || delayed.length > 0) {
      for (const text of texts.values()) yield* text.end()
      texts.clear()
      waitingForText = undefined
      for (const queued of delayed.splice(0)) yield* accept(queued)
    }
  }
  try {
    for await (const chunk of chunks) {
      if ('index' in chunk) {
        yield* accept(chunk)
        continue
      }
      yield* settle()
      if (chunk.type === 'usage') {
        yield chunk
        continue
      }
      if (chunk.reason.kind === 'stop' && blocks.length === 0) {
        yield { type: 'finish', reason: { kind: 'error', failure: {
          code: EMPTY_RESPONSE_CODE, message: 'model returned a completed response with no content after think-tag parsing',
        } } }
        continue
      }
      const replay = chunk.replayState
      yield replay === undefined ? chunk : {
        ...chunk,
        replayState: {
          response: replay.response,
          blocks: blocks.map(block => block.kind === undefined
            ? replay.blocks?.[block.source]
            : block.kind === 'reasoning'
              ? { type: 'reasoning', thinkingSignature: 'reasoning_content' }
              : { type: 'text' }),
        },
      }
    }
  } catch (error) {
    yield* settle()
    throw error
  }
}
