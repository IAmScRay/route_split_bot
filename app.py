import asyncio
import base64
import json
import os
from io import BytesIO
from typing import Any

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from openai import AsyncOpenAI


dp = Dispatcher()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is empty")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is empty")

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# user_id -> { "М10": {"weight": 281.42, "stores": [...]} }
user_to_routes: dict[int, dict[str, dict[str, Any]]] = {}
user_to_workers: dict[int, int] = {}


def distribution_score(buckets: list[list[dict[str, Any]]]) -> float:
    sums = [sum(item["weight"] for item in bucket) for bucket in buckets]
    return max(sums) - min(sums)


def distribute_routes(route_entries: list[dict[str, Any]], workers_count: int) -> list[list[dict[str, Any]]]:
    if workers_count <= 0:
        raise ValueError("workers_count must be > 0")

    if not route_entries:
        return [[] for _ in range(workers_count)]

    total_items = len(route_entries)
    base_count = total_items // workers_count
    remainder = total_items % workers_count

    target_counts = [
        base_count + 1 if i < remainder else base_count
        for i in range(workers_count)
    ]

    # Step 1: greedy initial distribution
    buckets = [[] for _ in range(workers_count)]
    bucket_sums = [0.0] * workers_count
    bucket_counts = [0] * workers_count

    entries = sorted(route_entries, key=lambda x: x["weight"], reverse=True)

    for entry in entries:
        best_worker = None
        best_key = None

        for i in range(workers_count):
            if bucket_counts[i] >= target_counts[i]:
                continue

            candidate_key = (bucket_sums[i], bucket_counts[i])

            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_worker = i

        if best_worker is None:
            raise RuntimeError("No available worker found")

        buckets[best_worker].append(entry)
        bucket_sums[best_worker] += entry["weight"]
        bucket_counts[best_worker] += 1

    # Step 2: local optimization
    improved = True
    while improved:
        improved = False
        current_score = distribution_score(buckets)

        # Try moving one route from one bucket to another
        for i in range(workers_count):
            for j in range(workers_count):
                if i == j:
                    continue

                if len(buckets[j]) >= target_counts[j]:
                    continue
                if len(buckets[i]) <= 0:
                    continue
                if len(buckets[i]) - 1 < target_counts[i] - 1:
                    continue

                for item in buckets[i]:
                    new_buckets = [bucket[:] for bucket in buckets]
                    new_buckets[i].remove(item)
                    new_buckets[j].append(item)

                    # respect target counts exactly
                    if len(new_buckets[i]) > target_counts[i] or len(new_buckets[j]) > target_counts[j]:
                        continue

                    if len(new_buckets[i]) < total_items // workers_count:
                        continue

                    new_score = distribution_score(new_buckets)
                    if new_score < current_score:
                        buckets = new_buckets
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

        if improved:
            continue

        # Try swapping one route between two buckets
        for i in range(workers_count):
            for j in range(i + 1, workers_count):
                for item_i in buckets[i]:
                    for item_j in buckets[j]:
                        new_buckets = [bucket[:] for bucket in buckets]

                        idx_i = new_buckets[i].index(item_i)
                        idx_j = new_buckets[j].index(item_j)

                        new_buckets[i][idx_i] = item_j
                        new_buckets[j][idx_j] = item_i

                        new_score = distribution_score(new_buckets)
                        if new_score < current_score:
                            buckets = new_buckets
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    return buckets


def normalize_route(route: str | None) -> str | None:
    if not route:
        return None

    route = route.strip().upper().replace("M", "М").replace(" ", "")
    if route.startswith("М") and route[1:].isdigit():
        return route

    if route.endswith("М") and route[:-1].isdigit():
        return f"М{route[:-1]}"

    digits = "".join(ch for ch in route if ch.isdigit())
    if digits:
        return f"М{digits}"

    return None


def validate_parsed_sheet(data: dict[str, Any]) -> tuple[bool, str]:
    route = normalize_route(data.get("route"))
    stores = data.get("stores", [])
    total = data.get("total_route_weight")

    if not route:
        return False, "Не вдалося розпізнати номер маршруту."

    if not isinstance(stores, list) or len(stores) == 0:
        return False, "Не вдалося розпізнати магазини на фото."

    cleaned_stores = []
    weights_sum = 0.0

    for store in stores:
        store_id = store.get("store_id")
        store_name = (store.get("store_name") or "").strip()
        total_weight = store.get("total_weight")

        if store_id is None or total_weight is None:
            continue

        try:
            total_weight = round(float(total_weight), 2)
        except (TypeError, ValueError):
            continue

        if total_weight <= 0:
            continue

        cleaned_stores.append(
            {
                "store_id": int(store_id),
                "store_name": store_name,
                "total_weight": total_weight,
            }
        )
        weights_sum += total_weight

    if not cleaned_stores:
        return False, "Модель не змогла надійно виділити вагу магазинів."

    if total is None:
        total = weights_sum
    else:
        try:
            total = round(float(total), 2)
        except (TypeError, ValueError):
            total = weights_sum

    # If model total is suspicious, trust the sum of extracted stores.
    if abs(total - weights_sum) > 1.0:
        total = round(weights_sum, 2)

    data["route"] = route
    data["stores"] = cleaned_stores
    data["total_route_weight"] = round(total, 2)

    return True, ""


async def parse_route_sheet_with_openai(image_bytes: bytes) -> dict[str, Any]:
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "route": {
                "type": ["string", "null"],
                "description": "Маршрут у форматі М<number>, наприклад М10",
            },
            "stores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "store_id": {"type": ["integer", "null"]},
                        "store_name": {"type": ["string", "null"]},
                        "total_weight": {"type": ["number", "null"]},
                    },
                    "required": ["store_id", "store_name", "total_weight"],
                },
            },
            "total_route_weight": {"type": ["number", "null"]},
            "needs_review": {"type": "boolean"},
            "notes": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["route", "stores", "total_route_weight", "needs_review", "notes"],
    }

    instructions = (
        "You extract route-sheet data from a Ukrainian delivery sheet photo. "
        "Return only visible information from the image. "
        "Important rules: "
        "1) Use the value after 'Всього замовлено' for each store, not the value after 'Ваговий'. "
        "2) There may be multiple order blocks on one photo. Extract all visible stores. "
        "3) Normalize route to the format 'М<number>' using Cyrillic М, for example 'М10'. "
        "4) total_route_weight must equal the sum of extracted store total_weight values when possible. "
        "5) If something is uncertain, set needs_review=true and mention it in notes. "
        "6) Do not invent missing stores or weights."
    )

    response = await openai_client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {
                "role": "system",
                "content": [
                    {"type": "input_text", "text": instructions}
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Parse this route sheet photo into the required JSON schema.",
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{image_b64}",
                        "detail": "high",
                    },
                ],
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "strict": True,
                "name": "route_sheet",
                "schema": schema,
            }
        },
        max_output_tokens=1200,
    )

    return json.loads(response.output_text)


async def download_largest_photo(bot: Bot, message: Message) -> bytes:
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)

    buffer = BytesIO()
    await bot.download(file, destination=buffer)
    return buffer.getvalue()


@dp.message(CommandStart())
async def start_handler(message: Message):
    await message.bot.send_message(
        chat_id=message.chat.id,
        text="👋 <b>Вітаю!</b>\n\nДаний бот допоможе розподілити маршрути на потрібну кількість працівників, "
             "автоматично підсумовуючи вагу кожного магазину у маршруті.\n\n"
             "Є два режими роботи:\n"
             "• <b>текстовий</b>: вкажіть номер маршруту та вагу для кожного магазину, після чого просто відправте повідомлення боту\n"
             "• <b>фото-режим</b>: надішліть боту фото маршрутного аркуша, й він автоматично розпізнає магазини та збереже маршрут\n\n"
             "Приклад повідомлення опису маршруту:\n"
             "<code>М24 68 32 105 44</code>\n\n"
             "Після того, як ви зберегли дані усіх маршрутів, вкажіть цифрою кількість працівників для завершення розподілу.",
        parse_mode=ParseMode.HTML
    )

    await message.delete()


@dp.message(F.photo)
async def photo_handler(message: Message, bot: Bot):
    user_id = message.from_user.id
    user_to_routes.setdefault(user_id, {})

    wait_msg = await message.answer("🔎 Аналізую фото...")

    try:
        image_bytes = await download_largest_photo(bot, message)
        parsed = await parse_route_sheet_with_openai(image_bytes)

        ok, error_text = validate_parsed_sheet(parsed)
        if not ok:
            await wait_msg.edit_text(
                f"❌ {error_text}\n\nСпробуйте ще раз: бажано фото зверху, без сильного нахилу та з усім листом у кадрі."
            )
            return

        route = parsed["route"]
        total_weight = parsed["total_route_weight"]
        stores = parsed["stores"]
        needs_review = parsed.get("needs_review", False)
        notes = parsed.get("notes", [])

        user_to_routes[user_id][route] = {
            "weight": total_weight,
            "stores": stores,
        }

        text = (
            f"✅ Маршрут <b>{route}</b> збережено\n"
            f"Загальна вага: <b>{total_weight:.2f} кг</b>\n"
            f"Магазинів знайдено: <b>{len(stores)}</b>\n\n"
        )

        for store in stores:
            store_name = f" {store['store_name']}" if store["store_name"] else ""
            text += (
                f"• <b>{store['store_id']}</b>{store_name}: "
                f"<b>{store['total_weight']:.2f} кг</b>\n"
            )

        if needs_review and notes:
            text += "\n⚠️ Потрібна перевірка:\n"
            for note in notes[:5]:
                text += f"• {note}\n"

        text += f"\nК-сть збережених маршрутів: <b>{len(user_to_routes[user_id])}</b>"

        await wait_msg.edit_text(text, parse_mode=ParseMode.HTML)

    except Exception as e:
        await wait_msg.edit_text(
            "❌ Не вдалося обробити фото.\n"
            f"<code>{str(e)[:500]}</code>",
            parse_mode=ParseMode.HTML,
        )


@dp.callback_query(F.data == "proceed")
async def split_proceed_handler(query: CallbackQuery):
    await query.answer()

    user_id = query.from_user.id

    route_map = user_to_routes.get(user_id, {})
    entries = [
        {"route": route, "weight": data["weight"]}
        for route, data in route_map.items()
    ]

    result = distribute_routes(entries, user_to_workers[user_id])

    text = "🧮 Результати розподілу:\n\n"
    for i, bucket in enumerate(result, start=1):
        bucket_total = sum(item["weight"] for item in bucket)
        text += f"Працівник <b>№{i}</b> ({bucket_total:.2f} кг):\n"

        for item in bucket:
            text += f"• <i>{item['route']}</i> (<b>{item['weight']:.2f} кг</b>)\n"

        if i != len(result):
            text += "\n–––––\n\n"

    await query.message.edit_text(text, parse_mode=ParseMode.HTML)

    user_to_workers.pop(user_id, None)
    user_to_routes.pop(user_id, None)


@dp.callback_query(F.data == "cancel")
async def split_cancel_handler(query: CallbackQuery):
    await query.answer()

    user_id = query.from_user.id
    user_to_workers.pop(user_id, None)

    await query.message.edit_text(
        "☝️ Операцію відмінено!\n\n"
        "За потреби можете надіслати додаткові фото, або ж додати чи видалити маршрут вручну."
    )


@dp.message(F.text.isdigit())
async def workers_count_handler(message: Message):
    user_id = message.from_user.id

    workers = int(message.text)
    if workers <= 0:
        await message.bot.send_message(
            chat_id=message.chat.id,
            text="❗️ Кількість працівників має бути 1 або більше."
        )
        await message.delete()
        return

    user_routes = user_to_routes.get(user_id, {})

    if not user_routes:
        await message.bot.send_message(
            chat_id=message.chat.id,
            text=
            "⚠️ У вас ще немає жодного збереженого маршруту<b>!</b>\n\n"
            "Спочатку надішліть фото маршрутного листа або введіть дані вручну.",
            parse_mode=ParseMode.HTML
        )
        await message.delete()
        return

    user_to_workers[user_id] = workers

    total_weight = 0.0
    text = "☝️ Перевірте маршрути перед розподілом:\n\n"

    for route, data in sorted(user_routes.items()):
        text += f"• {route}: <b>{data['weight']:.2f} кг</b>\n"
        total_weight += data["weight"]

    text += f"\n💪 Загальна вага: <b>{total_weight:.2f} кг</b>\n"
    text += f"Середня вага на кожного з <b>{workers}</b> працівників: ~<b>{(total_weight / workers):.2f} кг</b>\n\n"
    text += "Якщо все вірно — натисніть ✅\n"
    text += "Якщо потрібні доповнення чи виправлення - натисніть ❌"

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="✅", callback_data="proceed"),
                InlineKeyboardButton(text="❌", callback_data="cancel"),
            ]
        ]
    )

    await message.bot.send_message(
        chat_id=message.chat.id,
        text=text,
        reply_markup=keyboard,
        parse_mode=ParseMode.HTML
    )
    await message.delete()


@dp.message(~F.text.startwith("/") & F.text.startswith("М"))
async def route_text_handler(message: Message):
    user_id = message.from_user.id
    if user_id not in user_to_routes:
        user_to_routes[user_id] = {}

    parts = message.text.split(" ")
    route_str = parts[0]

    route_weight = 0
    for weight in parts[1::]:
        try:
            weight = int(float(weight))
        except ValueError:
            continue

        route_weight += weight

    user_to_routes[user_id][route_str] = {
        "weight": route_weight,
        "stores": []
    }

    await message.delete()
    await message.bot.send_message(
        chat_id=message.chat.id,
        text=
        f"✅ Маршрут <b>{route_str}</b> успішно збережено!\n"
        f"Загальна вага: ~<b>{route_weight} кг</b>\n\n"
        f"К-сть збережених маршрутів: <b>{len(user_to_routes[user_id])}</b>",
        parse_mode=ParseMode.HTML
    )


@dp.message(F.text.startswith("-М"))
async def route_delete_handler(message: Message):
    user_id = message.from_user.id

    if user_id not in user_to_routes:
        await message.bot.send_message(
            chat_id=message.chat.id,
            text=
            "⚠️ Ви ще не додавали жодного маршруту – видаляти нічого<b>!</b>\n\n"
            "Спочатку додайте маршрут за допомогою фото, або ж надішліть дані вручну.",
            parse_mode=ParseMode.HTML
        )
        await message.delete()

        return

    route = message.text[1::]
    if route not in user_to_routes[user_id]:
        await message.bot.send_message(
            chat_id=message.chat.id,
            text=
            f"🤨 Маршрут <b>{route}</b> відсутній у списку збережених маршрутів!\n\n"
            f"Впевніться у тому, що намаєтесь видалити саме потрібний вам маршрут.",
            parse_mode=ParseMode.HTML
        )
        await message.delete()

        return

    user_to_routes[user_id].pop(route, None)

    routes_left = len(user_to_routes[user_id])
    await message.bot.send_message(
        chat_id=message.chat.id,
        text=
        f"🗑️ Маршрут <b>{route}</b> успішно видалено!\n\n"
        f"Кількість збережених маршрутів: <b>{routes_left}</b>",
        parse_mode=ParseMode.HTML
    )

    await message.delete()


@dp.message()
async def fallback_handler(message: Message):
    await message.bot.send_message(
        chat_id=message.chat.id,
        text=
        "❌ Підтримуються команда <code>/start</code>, фото маршрутного аркуша або текстові дані маршруту наступного формату:\n"
        "<code>М24 68 32 105 44</code>\n\n"
        "Для видалення маршруту введіть маршрут з дефісом (мінусом) попереду, от як <code>-М24</code>",
        parse_mode=ParseMode.HTML,
    )
    await message.delete()


async def main():
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())