from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


TOPICS = ("conversation", "qa", "math", "logic", "coding", "commands")
SPLIT_NAMES = ("train", "validation", "test")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip()


def _record(question: str, answer: str) -> str:
    return f"کاربر: {_normalize(question)} پاسخ: {_normalize(answer)}"


def _examples(seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    rows: list[tuple[str, str]] = []

    # Short, varied Persian conversations. Names and topics are fictional
    # placeholders used only to make the smoke-test corpus less repetitive.
    names = ["آرمان", "بهار", "نیما", "رها", "سام", "مینا", "کیان", "ترانه", "پارسا", "هلیا", "مانی", "نازنین"]
    moods = ["خوب", "کمی خسته", "امیدوار", "آرام", "کنجکاو", "مشغول", "خوشحال", "فکرکننده"]
    needs = [
        ("یک تمرین کوتاه می‌خواهم", "حتماً؛ یک کار کوچک انتخاب کن و آن را به دو گام ساده تقسیم کن."),
        ("برای شروع برنامه‌نویسی راهنمایی می‌خواهم", "از یک مثال کوتاه شروع کن، آن را اجرا کن و بعد یک بخشش را تغییر بده."),
        ("می‌خواهم چیزی را بهتر یاد بگیرم", "موضوع را به قسمت‌های کوچک تقسیم کن و پس از هر قسمت یک پرسش از خودت بپرس."),
        ("دنبال یک ایده برای نوشتن هستم", "یک مکان خیالی انتخاب کن و در دو جمله بگو آنجا چه چیزی تغییر کرده است."),
        ("می‌خواهم مسئله‌ای را مرتب کنم", "واقعیت‌های معلوم را جدا بنویس و بعد فقط یک گام بعدی را انتخاب کن."),
        ("یک سؤال ساده دارم", "بپرس؛ اگر پاسخ در این تمرین معلوم باشد، آن را کوتاه توضیح می‌دهم."),
        ("برای استراحت کوتاه پیشنهادی می‌خواهم", "چند لحظه از صفحه فاصله بگیر و بعد با یک کار کوچک برگرد."),
        ("می‌خواهم یک فهرست بسازم", "سه مورد اصلی را بنویس و آن‌ها را به ترتیبی قرار بده که برایت مناسب است."),
    ]
    greetings = ["سلام", "درود", "وقت بخیر", "سلام، حالت چطور است؟"]
    for name in names:
        for mood in moods:
            for greeting in greetings:
                need, answer = rng.choice(needs)
                question = f"{greeting} من {name} هستم و امروز {mood} هستم؛ {need}."
                reply = f"سلام {name}. {answer}"
                rows.append(("conversation", _record(question, reply)))

    # Questions about an explicitly imaginary training table, not claims about
    # real-world facts. Every answer follows from the locally defined table.
    objects = ["مهره", "بادبادک", "کتابک", "گوی", "قاب", "پرچم", "سنگ", "جعبه", "برگ", "نشان", "کاشی", "چراغ", "توپ", "نقشه", "فنجان", "مداد", "قایق", "ستاره", "دکمه", "چرخ", "ابر", "زنگ", "کارت", "صدف"]
    colors = ["آبی", "سبز", "زرد", "بنفش", "نارنجی", "قرمز"]
    shapes = ["گرد", "چهارگوش", "سه‌گوش", "بیضی"]
    q_templates = [
        "در جدول خیالی تمرین، رنگ {item} شماره {index} چیست؟",
        "طبق جدول ساختگی این تمرین، شکل {item} شماره {index} را بگو.",
        "در بازی فرضی، {item} شماره {index} چه رنگی دارد؟",
        "اگر فقط به جدول همین تمرین نگاه کنیم، شکل {item} شماره {index} چیست؟",
    ]
    for index in range(1, 181):
        item = objects[(index - 1) % len(objects)]
        color = colors[(index * 3 + seed) % len(colors)]
        shape = shapes[(index * 5 + seed) % len(shapes)]
        answers = (f"در جدول خیالی این تمرین، رنگ آن {color} است.", f"شکل ثبت‌شده در جدول تمرینی {shape} است.")
        rows.append(("qa", _record(q_templates[index % len(q_templates)].format(item=item, index=index), answers[index % 2])))

    # Small whole-number arithmetic with deterministic, checkable answers.
    math_templates = {
        "+": ["حاصل جمع {a} و {b} چند می‌شود؟", "لطفاً {a} را با {b} جمع کن.", "اگر {a} واحد و {b} واحد دیگر داشته باشیم، جمع آن‌ها چیست؟", "پاسخ تمرین جمع {a} + {b} را بگو."],
        "-": ["اختلاف {a} و {b} چند است؟", "از {a}، مقدار {b} را کم کن.", "تمرین تفریق: {a} منهای {b} را حساب کن.", "اگر {a} چیز داشته باشیم و {b} تا برداریم، چند تا می‌ماند؟"],
        "×": ["حاصل ضرب {a} در {b} چیست؟", "{a} گروه {b}تایی روی‌هم چندتا می‌شود؟", "تمرین ضرب {a} × {b} را حل کن.", "اگر {a} بار {b} را جمع کنیم، پاسخ چیست؟"],
    }
    for a in range(0, 30):
        for b in range(0, 30):
            operations: Iterable[tuple[str, int]] = (("+", a + b), ("-", a - b))
            for op, result in operations:
                if op == "-" and result < 0:
                    continue
                for template_index, template in enumerate(math_templates[op]):
                    question = template.format(a=a, b=b)
                    answer = f"پاسخ {a} {op} {b} برابر {result} است."
                    # The arithmetic statement in the answer matches the
                    # requested operation, including subtraction prompts.
                    rows.append(("math", _record(question, answer)))
            if 0 <= a <= 12 and 0 <= b <= 12:
                result = a * b
                for template in math_templates["×"]:
                    rows.append(("math", _record(template.format(a=a, b=b), f"پاسخ {a} ضربدر {b} برابر {result} است.")))

    # A deliberately small fictional rule system; the conclusion is explicit.
    logic_templates = [
        "در بازی فرضی، هر {group} عضو {class_name} است و هر عضو {class_name} نشان {badge} می‌گیرد. آیا {group} نشان {badge} می‌گیرد؟",
        "قانون تمرینی می‌گوید هر {group} به گروه {class_name} تعلق دارد؛ همه اعضای {class_name} نشان {badge} دارند. نتیجه برای {group} چیست؟",
        "اگر در این دنیای خیالی هر {group} از نوع {class_name} باشد و هر {class_name} نشان {badge} داشته باشد، آیا {group} نشان دارد؟",
    ]
    groups = ["روباهک", "قایقک", "ابرک", "ستاره‌رو", "سنگ‌ریز", "پرنده‌نما", "چراغ‌بان", "رنگینک", "ماه‌پیما", "دانه‌ریز", "کتاب‌دوست", "بادگرد"]
    classes = ["رده آ", "رده ب", "رده پ", "رده ت", "رده ث", "رده ج"]
    badges = ["دایره", "مربع", "ستاره", "برگ", "موج", "ماه"]
    for index in range(180):
        group = groups[index % len(groups)]
        class_name = classes[(index // len(groups)) % len(classes)]
        badge = badges[(index * 7 + seed) % len(badges)]
        template = logic_templates[index % len(logic_templates)]
        rows.append(("logic", _record(template.format(group=group, class_name=class_name, badge=badge), f"بله؛ طبق دو قانون فرضی، {group} عضو {class_name} است و نشان {badge} می‌گیرد.")))

    # Beginner Python examples: code is illustrative text and is never run.
    coding_questions = [
        "در پایتون برای نمایش عدد {n} یک دستور ساده بنویس.",
        "یک خط کد پایتون بده که متن «سلام» را چاپ کند.",
        "دستور print در این نمونه چه چیزی نشان می‌دهد: print({n} + {m})؟",
        "برای ساخت متغیری به نام score با مقدار {n} چه خطی می‌نویسی؟",
        "در کد ساده زیر نتیجه چیست: for i in range({n}): print(i)؟",
        "برای بررسی این‌که {n} از {m} بزرگ‌تر است، شرط پایتون را بنویس.",
    ]
    for n in range(0, 40):
        m = (n * 7 + 3) % 17
        question = coding_questions[n % len(coding_questions)].format(n=n, m=m)
        if n % len(coding_questions) == 0:
            answer = f"برای چاپ این عدد می‌توان نوشت: print({n})؛ این نمونه فقط متن کد است و اجرا نمی‌شود."
        elif n % len(coding_questions) == 1:
            answer = 'یک نمونه کوتاه: print("سلام")؛ اینجا کد فقط توضیح داده می‌شود.'
        elif n % len(coding_questions) == 2:
            answer = f"حاصل عبارت {n} + {m} برابر {n + m} است؛ دستور print همین مقدار را نمایش می‌دهد."
        elif n % len(coding_questions) == 3:
            answer = f"خط نمونه این است: score = {n}؛ سپس متغیر score مقدار {n} را نگه می‌دارد."
        elif n % len(coding_questions) == 4:
            answer = f"حلقه اعداد صفر تا {max(0, n - 1)} را به‌ترتیب نمایش می‌دهد؛ این توضیح اجرای واقعی نیست."
        else:
            answer = f"شرط نمونه چنین است: {n} > {m}؛ در این مورد نتیجه {str(n > m).lower()} است."
        # Multiple Persian phrasings keep examples distinct while preserving
        # the same beginner-level concept.
        for phrasing in ("", " لطفاً پاسخ را کوتاه بنویس.", " فقط یک نمونه مقدماتی می‌خواهم."):
            rows.append(("coding", _record(question + phrasing, answer)))

    # Persian command descriptions are acknowledged as simulated, not executed.
    commands = ["باز کردن فهرست", "نمایش ساعت", "ساختن یادداشت", "پاک کردن یک سطر آزمایشی", "جست‌وجوی واژه در متن", "مرتب کردن فهرست", "شمردن واژه‌ها", "ذخیره پیش‌نویس", "بستن پنجره نمونه", "نمایش راهنما", "ساختن پوشه نمونه", "بررسی وضعیت تمرین", "تغییر عنوان یادداشت", "افزودن مورد به فهرست", "خواندن یک جمله", "محاسبه مجموع اعداد", "نمایش نام فایل", "یافتن کوتاه‌ترین واژه", "تبدیل حروف به کوچک", "مرتب کردن سه عدد"]
    command_forms = ["دستور فارسی: {command}.", "لطفاً در محیط تمرینی {command}.", "در این شبیه‌سازی می‌خواهم {command}.", "در فهرست فرضی، کار «{command}» را درخواست می‌کنم."]
    for index, command in enumerate(commands):
        for form_index, form in enumerate(command_forms):
            question = form.format(command=command)
            answer_forms = [
                f"در این نمونه، درخواست «{command}» ثبت شد؛ هیچ عملیات واقعی اجرا نشده است.",
                f"این فقط شبیه‌سازی متنی است: گام فرضی برای «{command}» آماده است.",
                f"توضیح آزمایشی: «{command}» یعنی یک کار پیشنهادی؛ این پاسخ به دستگاهی دسترسی ندارد.",
            ]
            rows.append(("commands", _record(question, answer_forms[(index + form_index) % len(answer_forms)])))

    return rows


def _hash_key(seed: int, topic: str, text: str) -> str:
    return hashlib.sha256(f"{seed}\0{topic}\0{text}".encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_dataset(output_dir: Path, seed: int = 20261121) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    by_topic: dict[str, list[tuple[str, str]]] = {topic: [] for topic in TOPICS}
    duplicates_removed = 0
    for topic, text in _examples(seed):
        normalized = _normalize(text)
        if len(normalized) < 32 or len(set(normalized)) < 9:
            duplicates_removed += 1
            continue
        key = normalized.casefold()
        if key in seen:
            duplicates_removed += 1
            continue
        seen.add(key)
        by_topic[topic].append((topic, normalized))

    splits: dict[str, list[tuple[str, str]]] = {name: [] for name in SPLIT_NAMES}
    for topic in TOPICS:
        ordered = sorted(by_topic[topic], key=lambda row: _hash_key(seed, topic, row[1]))
        count = len(ordered)
        train_end = int(count * 0.80)
        validation_end = int(count * 0.90)
        splits["train"].extend(ordered[:train_end])
        splits["validation"].extend(ordered[train_end:validation_end])
        splits["test"].extend(ordered[validation_end:])

    per_split_topics: dict[str, dict[str, int]] = {}
    split_counts: dict[str, int] = {}
    file_hashes: dict[str, str] = {}
    split_text: dict[str, list[str]] = {}
    for split_name in SPLIT_NAMES:
        rows = sorted(splits[split_name], key=lambda row: _hash_key(seed, row[0], row[1]))
        if not rows:
            raise RuntimeError(f"Synthetic {split_name} split is empty")
        if len({text.casefold() for _, text in rows}) != len(rows):
            raise RuntimeError(f"Duplicate rows remain in synthetic {split_name} split")
        split_text[split_name] = [text for _, text in rows]
        per_split_topics[split_name] = dict(Counter(topic for topic, _ in rows))
        split_counts[split_name] = len(rows)
        path = output_dir / f"{split_name}.txt"
        path.write_text("\n".join(split_text[split_name]) + "\n", encoding="utf-8")
        file_hashes[f"sha256_{split_name}"] = _sha256(path)

    # The same synthetic example must never appear in more than one split.
    all_rows = [text.casefold() for values in split_text.values() for text in values]
    if len(all_rows) != len(set(all_rows)):
        raise RuntimeError("Synthetic split leakage detected: an example occurs in multiple splits")
    if set(per_split_topics["train"]) != set(TOPICS):
        raise RuntimeError("One or more topics are absent from the training split")
    for split_name in SPLIT_NAMES:
        missing_topics = sorted(set(TOPICS) - set(per_split_topics[split_name]))
        if missing_topics:
            raise RuntimeError(f"{split_name} split is missing topics: {', '.join(missing_topics)}")

    topic_counts = {topic: len(by_topic[topic]) for topic in TOPICS}
    manifest: dict[str, object] = {
        "project": "Monster-1",
        "dataset_name": "Monster-1 Persian synthetic smoke-test corpus",
        "manifest_version": 1,
        "mode": "synthetic_test",
        "is_synthetic": True,
        "synthetic": True,
        "real_data": False,
        "pretrained_model_used": False,
        "external_corpus_used": False,
        "external_api_used": False,
        "generation_method": "Deterministic local Persian templates and finite arithmetic, fictional-table logic, beginner Python, conversation, and simulated-command combinations; no network or model calls.",
        "generation_seed": seed,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_method": "Per-topic deterministic SHA-256 ordering, 80% train / 10% validation / 10% test; exact normalized duplicate removal before splitting.",
        "sample_count": len(all_rows),
        "sample_counts": split_counts,
        "topic_sample_counts": topic_counts,
        "split_topic_sample_counts": per_split_topics,
        "duplicates_removed": duplicates_removed,
        "sha256_train": file_hashes["sha256_train"],
        "sha256_validation": file_hashes["sha256_validation"],
        "sha256_test": file_hashes["sha256_test"],
        "quality_note": "Synthetic examples are for pipeline smoke testing only; they are not a real or licensed Persian corpus and do not establish model quality.",
    }
    manifest_path = output_dir / "corpus_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a local Persian synthetic test corpus; makes no network calls")
    parser.add_argument("--output-dir", type=Path, default=Path("data/synthetic_test"))
    parser.add_argument("--seed", type=int, default=20261121)
    args = parser.parse_args()
    manifest = generate_dataset(args.output_dir, seed=args.seed)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
