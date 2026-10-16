/** Native stream activity remains observable while tag parsing buffers interleaved chunks. */
import { expect, it, vi } from 'vitest'
import { BlockAssembler } from '@deepseek-ai/dsh-llm'
import type { AssistantMessage, AssistantMessageEvent, StreamOptions } from '@earendil-works/pi-ai'
import { AssistantMessageEventStream } from '@earendil-works/pi-ai/utils/event-stream'
import { PiAiAdapter } from '../src/adapter.ts'
import { resolveProfiles } from '../src/config.ts'
import { memoryAuth } from './auth-double.ts'

function nativeFixture(prefix: string) {
  const profiles = resolveProfiles({ gateway: {
    apiKeyEnv: 'PI_TEST_KEY', api: 'openai-completions', baseURL: 'https://example.test/v1', streamIdleTimeoutMs: 1000,
    models: [{ id: 'tagged', parseThinkTags: true }],
  } })
  const profile = profiles.get('gateway')
  if (profile?.piProvider === undefined) throw new Error('fixture gateway did not resolve')
  const events = new AssistantMessageEventStream()
  const started = Promise.withResolvers<AbortSignal>()
  const stream = (_model: unknown, _context: unknown, options?: StreamOptions) => {
    if (options?.signal === undefined) throw new Error('adapter did not supply a signal')
    started.resolve(options.signal)
    options.signal.addEventListener('abort', () => {
      const aborted = { ...partial, stopReason: 'aborted' as const }
      events.push({ type: 'error', reason: 'aborted', error: aborted })
      events.end(aborted)
    }, { once: true })
    return events
  }
  profiles.set('gateway', { ...profile, piProvider: { ...profile.piProvider, stream, streamSimple: stream } })
  const adapter = new PiAiAdapter({
    profiles: () => profiles, resolveApiKey: () => Promise.resolve('fixture-key'), auth: memoryAuth(),
  })
  const partial: AssistantMessage = {
    role: 'assistant', api: 'openai-completions', provider: 'gateway', model: 'tagged', timestamp: 0, stopReason: 'toolUse',
    content: [
      { type: 'text', text: prefix }, { type: 'toolCall', id: 'call-1', name: 'echo', arguments: {} },
    ],
    usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
  }
  return { adapter, events, partial, started }
}

it.each(['<thi', '<think>'])('counts native activity while %s has no normalized output', async (prefix) => {
  const { adapter, events, partial, started } = nativeFixture(prefix)
  const progress: AssistantMessageEvent[] = [
    { type: 'text_start', contentIndex: 0, partial },
    { type: 'text_delta', contentIndex: 0, delta: prefix, partial },
    { type: 'toolcall_start', contentIndex: 1, partial },
    ...['', '{', '"', 'x', '"', ':', '1', '}'].map((delta): AssistantMessageEvent => ({ type: 'toolcall_delta', contentIndex: 1, delta, partial })),
  ]
  const assembler = new BlockAssembler()
  vi.useFakeTimers()
  const reading = (async () => {
    try {
      for await (const chunk of adapter.stream({ provider: 'gateway', model: 'tagged', messages: [] })) assembler.push(chunk)
      return { kind: 'done' }
    } catch (error) {
      return { kind: 'error', error }
    }
  })()
  try {
    const signal = await started.promise
    for (const event of progress) {
      events.push(event)
      await vi.advanceTimersByTimeAsync(750)
      expect(signal.aborted).toBe(false)
    }
    expect(assembler.blocks()).toEqual([])
    const suffix = prefix === '<thi' ? 'nk>plan</think>answer' : 'plan</think>answer'
    events.push({ type: 'text_delta', contentIndex: 0, delta: suffix, partial })
    events.push({ type: 'text_end', contentIndex: 0, content: prefix + suffix, partial })
    events.push({ type: 'toolcall_end', contentIndex: 1, toolCall: { type: 'toolCall', id: 'call-1', name: 'echo', arguments: { x: 1 } }, partial })
    events.push({ type: 'done', reason: 'toolUse', message: partial })
    events.end(partial)
    expect(await reading).toEqual({ kind: 'done' })
    expect(assembler.blocks()).toEqual([
      { type: 'reasoning', text: 'plan' }, { type: 'text', text: 'answer' },
      { type: 'tool-call', id: 'call-1', name: 'echo', arguments: '{"x":1}' },
    ])
  } finally {
    events.end(partial)
    await reading
    vi.useRealTimers()
  }
})

it('closes the native stream when a consumer returns from a normalized block', async () => {
  const { adapter, events, partial, started } = nativeFixture('<think>plan')
  const reading = (async () => {
    for await (const chunk of adapter.stream({ provider: 'gateway', model: 'tagged', messages: [] })) {
      expect(chunk).toMatchObject({ type: 'block-start', blockType: 'reasoning' })
      break
    }
  })()
  try {
    const signal = await started.promise
    events.push({ type: 'text_start', contentIndex: 0, partial })
    events.push({ type: 'text_delta', contentIndex: 0, delta: '<think>plan', partial })
    await reading
    expect(signal.aborted).toBe(true)
    expect((await events.result()).stopReason).toBe('aborted')
  } finally {
    events.end(partial)
    await reading
  }
})
