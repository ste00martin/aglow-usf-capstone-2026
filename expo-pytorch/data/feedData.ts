/**
 * feedData.ts
 *
 * Type definitions and Flickr8k feed items (Kaggle dataset).
 * Images are bundled via require() in the generated kaggleFeedItems.ts.
 * Re-generate with: python scripts/prepare_kaggle_feed.py
 */

import { KAGGLE_FEED_ITEMS } from './kaggleFeedItems';

export type FeedItem = {
  id: string;
  uri: string | number;  // number = require() bundled asset reference
  caption: string;
  tags: string[];
  source: 'kaggle' | 'dummy';
  kaggleId?: string;
};

export const FEED_ITEMS: FeedItem[] = KAGGLE_FEED_ITEMS;

export const REPORT_CATEGORIES = [
  'Inappropriate / NSFW',
  'Violence / Gore',
  'Spam or Misleading',
  'Hate Speech',
  'Irrelevant Content',
  'Other',
] as const;

export type ReportCategory = typeof REPORT_CATEGORIES[number];
