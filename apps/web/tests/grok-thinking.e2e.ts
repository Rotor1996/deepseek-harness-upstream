/** Grok reasoning remains independently expandable after a recorded tool round and reload. */
import { ReasoningEffortId } from '@deepseek-ai/dsh-llm'
import { readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { join } from 'node:path'
import { chromium, type Browser, type Page } from 'playwright'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import {
  captureStableAria, compareOrRefreshGolden, fixtureUserPrompts, launchWebScaffold,
  recordFixture, webSnapshotMode, type WebScaffold,
} from './scaffold.ts'
import { connectFreshWorkspace, expandOwningTurnProcess, newEnglishPage } from './support.ts'

const DIR = fileURLToPath(new URL('../../../snapshots/web/grok-thinking', import.meta.url))
const FIXTURE = join(DIR, 'session.v4.jsonl')
const MODE = webSnapshotMode()
const PROMPT = 'Find the smallest positive integer divisible by 7 that leaves remainder 3 when divided by 11 and remainder 5 when divided by 13. Record the result as one completed item using todo_write exactly once. Then reply exactly GROK_THINKING_READY. Do not call any other tool.'

describe('web e2e: Grok reasoning disclosure', () => {
  let scaffold: WebScaffold
  let browser: Browser
  let page: Page

  beforeAll(async () => {
    scaffold = await launchWebScaffold(MODE === 'record' ? {} : {
      replayFixture: FIXTURE, paceMs: 5,
      replayProviders: [{ id: 'grok-gateway', name: 'Grok gateway', models: [{
        id: 'grok-4.7', name: 'grok-4.7', contextWindow: 500_000, defaultMaxTokens: 2048,
        reasoningEfforts: ['low'], defaultReasoningEffort: 'low',
      }] }],
    })
    if (MODE === 'record') {
      if (!process.env.DSH_GROK_BASE_URL || !process.env.DSH_GROK_API_KEY) throw new Error('Grok recording needs DSH_GROK_BASE_URL and DSH_GROK_API_KEY')
      await scaffold.ctx.settings.update('llm-pi-ai', { providers: { 'grok-gateway': {
        displayName: 'Grok gateway', api: 'openai-completions', baseURL: process.env.DSH_GROK_BASE_URL,
        apiKeyEnv: 'DSH_GROK_API_KEY', models: [{ id: 'grok-4.7', contextWindow: 500_000,
          parseThinkTags: true, reasoningEfforts: { low: 'low' }, maxTokens: 2048 }],
      } } })
    }
    await scaffold.ctx.agentDefaultModel.saveSelection({ provider: 'grok-gateway', model: 'grok-4.7', reasoningEffort: ReasoningEffortId('low') })
    browser = await chromium.launch()
    page = await newEnglishPage(browser)
    await page.goto(scaffold.authenticatedUrl, { waitUntil: 'load' })
    await connectFreshWorkspace(page, scaffold.workspaceCwd)
  }, 120_000)

  afterAll(async () => {
    try { await browser?.close() } finally { await scaffold?.close() }
  })

  it('keeps reasoning out of the answer and expands it after reload', async () => {
    if (MODE !== 'record') expect(fixtureUserPrompts(await readFile(FIXTURE, 'utf8'))).toEqual([PROMPT])
    const settled = scaffold.whenTurnSettled()
    await page.locator('[data-composer-input]').first().fill(PROMPT)
    await page.locator('[data-composer-input]').first().press('Enter')
    const id = await settled
    const session = scaffold.ctx.sessions.get(id)!
    const blocks = session.deriveMessages().filter(message => message.role === 'assistant').flatMap(message => message.content)
    expect(blocks.filter(block => block.type === 'tool-call').map(block => block.name)).toEqual(['todo_write'])
    expect(blocks.some(block => block.type === 'reasoning' && block.text.length > 0)).toBe(true)
    expect(blocks.filter(block => block.type === 'text').map(block => block.text).join('')).not.toContain('<think>')
    await page.getByText('GROK_THINKING_READY', { exact: true }).waitFor()
    if (MODE === 'record') await recordFixture(scaffold, id, FIXTURE)
    await page.reload({ waitUntil: 'load' })
    await page.getByText('GROK_THINKING_READY', { exact: true }).waitFor()
    const reasoning = page.locator('[data-variant="think"]').first()
    await expandOwningTurnProcess(page, reasoning)
    await reasoning.locator('[aria-expanded]').first().click()
    await expect.poll(() => reasoning.getAttribute('data-expanded')).toBe('true')
    if (MODE !== 'record') await compareOrRefreshGolden(join(DIR, 'ui.expected.md'),
      await captureStableAria(page, '[class*="centerCol"]', scaffold.workspaceCwd), MODE)
    await reasoning.locator('[aria-expanded]').first().click()
    await expect.poll(() => reasoning.getAttribute('data-expanded')).toBeNull()
  })
})
