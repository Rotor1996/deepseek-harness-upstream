/** Streaming tag parsing, metadata alignment, and transport failure coverage. */
import { describe, expect, it } from 'vitest'
import { BlockAssembler, EMPTY_RESPONSE_CODE, ToolCallId } from '@deepseek-ai/dsh-llm'
import type { FinishReason, StreamChunk } from '@deepseek-ai/dsh-llm'
import { parseThinkTagStream } from '../src/think-tags.ts'
import { Config, resolveProfiles, type PiAiProviderProfile } from '../src/config.ts'

async function* source(chunks: StreamChunk[]): AsyncGenerator<StreamChunk> {
  yield* chunks
}

function textChunks(parts: string[]): StreamChunk[] {
  return [
    { type: 'block-start', index: 0, blockType: 'text' },
    ...parts.map((text): StreamChunk => ({ type: 'text-delta', index: 0, text })),
    { type: 'block-end', index: 0, block: { type: 'text', text: parts.join('') } },
  ]
}

async function collect(chunks: StreamChunk[]): Promise<{ assembler: BlockAssembler; chunks: StreamChunk[] }> {
  const result: StreamChunk[] = []
  for await (const chunk of parseThinkTagStream(source(chunks))) result.push(chunk)
  const assembler = new BlockAssembler()
  for (const chunk of result) assembler.push(chunk)
  return { assembler, chunks: result }
}

const finish: StreamChunk = {
  type: 'finish', reason: { kind: 'stop' },
  replayState: { response: { id: 'response' }, blocks: [{ type: 'text' }, { type: 'tool-call' }] },
}

function content(assembler: BlockAssembler) {
  return assembler.message({ provider: 'gateway', model: 'grok' }).content
}

describe('leading think tags', () => {
  const tagged = '<think>plan</think>answer'
  it.each(Array.from({ length: tagged.length + 1 }, (_, index) => index))('handles a split at byte %i', async (index) => {
    const { assembler } = await collect([...textChunks([tagged.slice(0, index), tagged.slice(index)]), finish])
    expect(content(assembler)).toEqual([{ type: 'reasoning', text: 'plan' }, { type: 'text', text: 'answer' }])
    expect(assembler.replayState).toEqual({ response: { id: 'response' }, blocks: [
      { type: 'reasoning', thinkingSignature: 'reasoning_content' }, { type: 'text' },
    ] })
  })

  it.each(['ordinary <think>literal</think>', '```xml\n<think>literal</think>\n```', '<thinking>literal</thinking>', '<thi', '', '<thinker>literal'])('preserves literal text %j', async (text) => {
    const { assembler } = await collect([...textChunks(text.split('')), finish])
    expect(content(assembler)).toEqual(text ? [{ type: 'text', text }] : [])
  })

  it.each(['<think>plan', '<think>plan</thi'])('keeps unclosed reasoning before a tool call: %s', async (text) => {
    const id = ToolCallId('call-1')
    const { assembler, chunks } = await collect([
      ...textChunks(text.split('')),
      { type: 'block-start', index: 1, blockType: 'tool-call' },
      { type: 'tool-call-delta', index: 1, id, name: 'echo', argumentsDelta: '{}' },
      { type: 'block-end', index: 1, block: { type: 'tool-call', id, name: 'echo', arguments: '{}' } },
      { type: 'usage', usage: { inputTokens: 3, outputTokens: 4, totalTokens: 7 } },
      { ...finish, reason: { kind: 'tool-calls' } },
    ])
    expect(content(assembler)).toEqual([
      { type: 'reasoning', text: text.slice(7) },
      { type: 'tool-call', id, name: 'echo', arguments: '{}' },
    ])
    expect(assembler.replayState?.blocks).toEqual([{ type: 'reasoning', thinkingSignature: 'reasoning_content' }, { type: 'tool-call' }])
    expect(chunks.at(-2)).toEqual({ type: 'usage', usage: { inputTokens: 3, outputTokens: 4, totalTokens: 7 } })
    expect(assembler.finish).toEqual({ kind: 'tool-calls' })
  })

  it.each([
    { prefix: '<', tail: 'a', expected: [{ type: 'text', text: '<a' }] },
    { prefix: '<thi', tail: 'nk>plan</think>answer', expected: [{ type: 'reasoning', text: 'plan' }, { type: 'text', text: 'answer' }] },
    { prefix: '<think>', tail: 'plan</think>answer', expected: [{ type: 'reasoning', text: 'plan' }, { type: 'text', text: 'answer' }] },
    { prefix: '<thi', tail: '', expected: [{ type: 'text', text: '<thi' }] },
  ])('keeps an interleaved tool after undecided text $prefix', async ({ prefix, tail, expected }) => {
    const id = ToolCallId('call-1')
    const { assembler } = await collect([
      { type: 'block-start', index: 0, blockType: 'text' },
      { type: 'text-delta', index: 0, text: prefix },
      { type: 'block-start', index: 1, blockType: 'tool-call' },
      { type: 'tool-call-delta', index: 1, id, name: 'echo', argumentsDelta: '{}' },
      { type: 'text-delta', index: 0, text: tail },
      { type: 'block-end', index: 0, block: { type: 'text', text: prefix + tail } },
      { type: 'block-end', index: 1, block: { type: 'tool-call', id, name: 'echo', arguments: '{}' } },
      { ...finish, reason: { kind: 'tool-calls' } },
    ])
    expect(content(assembler)).toEqual([...expected, { type: 'tool-call', id, name: 'echo', arguments: '{}' }])
    expect(assembler.replayState?.blocks?.map(block => (block as { type: string }).type))
      .toEqual(content(assembler).map(block => block.type))
  })

  it('emits tools before later answer text once reasoning has started', async () => {
    const id = ToolCallId('call-1')
    const { assembler } = await collect([
      { type: 'block-start', index: 0, blockType: 'text' },
      { type: 'text-delta', index: 0, text: '<think>plan</think>' },
      { type: 'block-start', index: 1, blockType: 'tool-call' },
      { type: 'tool-call-delta', index: 1, id, name: 'echo', argumentsDelta: '{}' },
      { type: 'text-delta', index: 0, text: 'answer' },
      { type: 'block-end', index: 0, block: { type: 'text', text: '<think>plan</think>answer' } },
      { type: 'block-end', index: 1, block: { type: 'tool-call', id, name: 'echo', arguments: '{}' } },
      { ...finish, reason: { kind: 'tool-calls' } },
    ])
    expect(content(assembler)).toEqual([
      { type: 'reasoning', text: 'plan' }, { type: 'tool-call', id, name: 'echo', arguments: '{}' }, { type: 'text', text: 'answer' },
    ])
    expect(assembler.replayState?.blocks).toEqual([
      { type: 'reasoning', thinkingSignature: 'reasoning_content' }, { type: 'tool-call' }, { type: 'text' },
    ])
  })

  it.each(['<think>', '<think></think>'])('rejects a successful response emptied by %s', async (text) => {
    const { assembler } = await collect([...textChunks([text]), finish])
    expect(content(assembler)).toEqual([])
    expect(assembler.finish).toMatchObject({ kind: 'error', failure: { code: EMPTY_RESPONSE_CODE } })
  })

  it.each<FinishReason>([
    { kind: 'max-tokens' },
    { kind: 'error', failure: { code: 'TRANSPORT', message: 'connection closed' } },
    { kind: 'aborted', failure: { code: 'ABORTED', message: 'cancelled' } },
  ])('preserves $kind for empty tag content', async (reason) => {
    const { assembler } = await collect([...textChunks(['<think></think>']), { type: 'finish', reason }])
    expect(content(assembler)).toEqual([])
    expect(assembler.finish).toEqual(reason)
  })

  it.each<FinishReason>([
    { kind: 'error', failure: { code: 'TRANSPORT', message: 'connection closed' } },
    { kind: 'aborted', failure: { code: 'ABORTED', message: 'cancelled' } },
  ])('releases buffered interleaved blocks before a $kind finish', async (reason) => {
    const id = ToolCallId('call-1')
    const { assembler } = await collect([
      { type: 'block-start', index: 0, blockType: 'text' },
      { type: 'text-delta', index: 0, text: '<thi' },
      { type: 'block-start', index: 1, blockType: 'tool-call' },
      { type: 'tool-call-delta', index: 1, id, name: 'echo', argumentsDelta: '{' },
      { type: 'usage', usage: { inputTokens: 3, outputTokens: 4, totalTokens: 7 } },
      { type: 'finish', reason },
    ])
    expect(content(assembler).map(block => block.type)).toEqual(['text', 'tool-call'])
    expect(assembler.interruptedBlocks()).toEqual([{ type: 'text', text: '<thi' }])
    expect(assembler.finish).toEqual(reason)
  })

  it('keeps a tool-only response after dropping an empty leading think section', async () => {
    const id = ToolCallId('call-1')
    const { assembler } = await collect([
      { type: 'block-start', index: 0, blockType: 'text' },
      { type: 'text-delta', index: 0, text: '<think>' },
      { type: 'block-start', index: 1, blockType: 'tool-call' },
      { type: 'tool-call-delta', index: 1, id, name: 'echo', argumentsDelta: '{}' },
      { type: 'block-end', index: 0, block: { type: 'text', text: '<think>' } },
      { type: 'block-end', index: 1, block: { type: 'tool-call', id, name: 'echo', arguments: '{}' } },
      { ...finish, reason: { kind: 'tool-calls' } },
    ])
    expect(content(assembler)).toEqual([{ type: 'tool-call', id, name: 'echo', arguments: '{}' }])
    expect(assembler.finish).toEqual({ kind: 'tool-calls' })
    expect(assembler.replayState?.blocks).toEqual([{ type: 'tool-call' }])
  })

  it('retains native reasoning metadata and remaps interleaved tool calls', async () => {
    const id = ToolCallId('call-1')
    const { assembler } = await collect([
      { type: 'block-start', index: 0, blockType: 'reasoning' },
      { type: 'reasoning-delta', index: 0, text: 'native' },
      { type: 'block-start', index: 1, blockType: 'text' },
      { type: 'text-delta', index: 1, text: '<think>tagged</think>answer' },
      { type: 'block-start', index: 2, blockType: 'tool-call' },
      { type: 'tool-call-delta', index: 2, id, name: 'echo', argumentsDelta: '{}' },
      { type: 'block-end', index: 0, block: { type: 'reasoning', text: 'native' } },
      { type: 'block-end', index: 1, block: { type: 'text', text: '<think>tagged</think>answer' } },
      { type: 'block-end', index: 2, block: { type: 'tool-call', id, name: 'echo', arguments: '{}' } },
      { type: 'finish', reason: { kind: 'tool-calls' }, replayState: { response: {}, blocks: [
        { type: 'reasoning', thinkingSignature: 'reasoning' }, { type: 'text' }, { type: 'tool-call' },
      ] } },
    ])
    expect(content(assembler).map(block => block.type)).toEqual(['reasoning', 'reasoning', 'text', 'tool-call'])
    expect(assembler.replayState?.blocks).toEqual([
      { type: 'reasoning', thinkingSignature: 'reasoning' }, { type: 'reasoning', thinkingSignature: 'reasoning_content' },
      { type: 'text' }, { type: 'tool-call' },
    ])
  })

  it('drops empty reasoning without losing the answer metadata', async () => {
    const { assembler } = await collect([...textChunks(['<think></think>answer <think>literal</think>']), finish])
    expect(content(assembler)).toEqual([{ type: 'text', text: 'answer <think>literal</think>' }])
    expect(assembler.replayState?.blocks).toEqual([{ type: 'text' }])
  })

  it('flushes partial closing tags on aborted finish', async () => {
    const { assembler } = await collect([
      { type: 'block-start', index: 0, blockType: 'text' },
      { type: 'text-delta', index: 0, text: '<think>plan</thi' },
      { type: 'finish', reason: { kind: 'aborted', failure: { code: 'ABORTED', message: 'cancelled' } } },
    ])
    expect(content(assembler)).toEqual([{ type: 'reasoning', text: 'plan</thi' }])
    expect(assembler.finish.kind).toBe('aborted')
  })

  it('flushes pending text and preserves a thrown transport failure', async () => {
    const failure = new Error('transport failed')
    async function* failed(): AsyncGenerator<StreamChunk> {
      yield { type: 'block-start', index: 0, blockType: 'text' }
      yield { type: 'text-delta', index: 0, text: '<thi' }
      throw failure
    }
    const iterator = parseThinkTagStream(failed())
    expect((await iterator.next()).value).toEqual({ type: 'block-start', index: 0, blockType: 'text' })
    expect((await iterator.next()).value).toEqual({ type: 'text-delta', index: 0, text: '<thi' })
    expect((await iterator.next()).value).toEqual({ type: 'block-end', index: 0, block: { type: 'text', text: '<thi' } })
    await expect(iterator.next()).rejects.toBe(failure)
  })
})

describe('model-scoped tag configuration', () => {
  it('defaults off and only enables the selected model', () => {
    const providers: Record<string, PiAiProviderProfile> = { gateway: { api: 'openai-completions', baseURL: 'https://example.test/v1', models: [
      { id: 'tagged', parseThinkTags: true }, { id: 'default' }, { id: 'disabled', parseThinkTags: false },
    ] } }
    expect(Config({ providers }).providers.get()).toHaveProperty('gateway.models.0.parseThinkTags', true)
    expect(resolveProfiles(providers).get('gateway')?.thinkTagModels).toEqual(new Set(['tagged']))
  })

  it('rejects opt-in on a different protocol', () => {
    expect(() => resolveProfiles({ gateway: { api: 'anthropic-messages', baseURL: 'https://example.test', models: [
      { id: 'tagged', parseThinkTags: true },
    ] } })).toThrow('parseThinkTags requires openai-completions')
  })

  it('supports installed model overrides', () => {
    expect(resolveProfiles({ deepseek: { modelOverrides: { 'deepseek-flash': { parseThinkTags: true } } } })
      .get('deepseek')?.thinkTagModels).toEqual(new Set(['deepseek-flash']))
  })
})
