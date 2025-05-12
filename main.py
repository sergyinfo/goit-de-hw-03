# Імпортуємо необхідні бібліотеки
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType, DateType

# 1. Ініціалізація SparkSession
spark = SparkSession.builder \
    .appName("Analyze User Purchases") \
    .getOrCreate()

# --- Крок 1: Завантаження та читання CSV-файлів ---
# Визначаємо шляхи до файлів
users_path = "users.csv"
purchases_path = "purchases.csv"
products_path = "products.csv"

# Читаємо файли, вказуючи наявність заголовка та автоматичне визначення схеми
try:
    users_df = spark.read.csv(users_path, header=True, inferSchema=True)
    purchases_df = spark.read.csv(purchases_path, header=True, inferSchema=True)
    products_df = spark.read.csv(products_path, header=True, inferSchema=True)

    print("Початкові дані завантажено:")
    print("Користувачі:")
    users_df.show(5)
    users_df.printSchema()
    print("Покупки:")
    purchases_df.show(5)
    purchases_df.printSchema()
    print("Продукти:")
    products_df.show(5)
    products_df.printSchema()

except Exception as e:
    print(f"Помилка при читанні CSV файлів: {e}")
    spark.stop()
    exit()


# --- Крок 2: Очищення даних (видалення рядків з пропущеними значеннями) ---
# Рахуємо рядки до очищення
initial_user_count = users_df.count()
initial_purchase_count = purchases_df.count()
initial_product_count = products_df.count()

# Видаляємо рядки, де будь-яке значення є NULL
users_df_cleaned = users_df.dropna()
purchases_df_cleaned = purchases_df.dropna()
products_df_cleaned = products_df.dropna()

# Рахуємо рядки після очищення
cleaned_user_count = users_df_cleaned.count()
cleaned_purchase_count = purchases_df_cleaned.count()
cleaned_product_count = products_df_cleaned.count()

print("\n--- Результати очищення даних ---")
print(f"Користувачі: Видалено {initial_user_count - cleaned_user_count} рядків з пропущеними значеннями.")
print(f"Покупки: Видалено {initial_purchase_count - cleaned_purchase_count} рядків з пропущеними значеннями.")
print(f"Продукти: Видалено {initial_product_count - cleaned_product_count} рядків з пропущеними значеннями.")

# --- Крок 3: Загальна сума покупок за кожною категорією продуктів ---

# Об'єднуємо покупки та продукти
purchases_with_products_df = purchases_df_cleaned.join(
    products_df_cleaned,
    purchases_df_cleaned["product_id"] == products_df_cleaned["product_id"],
    "inner" # Використовуємо inner join, щоб врахувати тільки покупки існуючих продуктів
).drop(products_df_cleaned["product_id"]) # Видаляємо дублікат колонки product_id

# Розраховуємо вартість кожної покупки (ціна * кількість)
purchases_with_total_cost = purchases_with_products_df.withColumn(
    "total_cost", F.col("quantity") * F.col("price")
)

# Групуємо за категорією та обчислюємо суму
category_spending_total_df = purchases_with_total_cost.groupBy("category") \
    .agg(F.sum("total_cost").alias("total_spending_per_category")) \
    .orderBy(F.col("total_spending_per_category").desc())

print("\n--- Крок 3: Загальна сума покупок за категоріями ---")
category_spending_total_df.show()


# --- Крок 4: Сума покупок за категоріями для вікової категорії 18-25 ---

# Фільтруємо користувачів за віком
users_18_25_df = users_df_cleaned.filter((F.col("age") >= 18) & (F.col("age") <= 25))

# Об'єднуємо відфільтрованих користувачів з покупками та продуктами
purchases_18_25_df = users_18_25_df.join(
    purchases_df_cleaned,
    users_18_25_df["user_id"] == purchases_df_cleaned["user_id"],
    "inner"
).select(purchases_df_cleaned["purchase_id"], purchases_df_cleaned["user_id"], purchases_df_cleaned["product_id"], purchases_df_cleaned["quantity"])

purchases_18_25_with_products_df = purchases_18_25_df.join(
    products_df_cleaned,
    purchases_18_25_df["product_id"] == products_df_cleaned["product_id"],
    "inner"
).drop(products_df_cleaned["product_id"])

# Розраховуємо вартість кожної покупки для цієї групи
purchases_18_25_with_total_cost = purchases_18_25_with_products_df.withColumn(
    "total_cost", F.col("quantity") * F.col("price")
)

# Групуємо за категорією та обчислюємо суму для групи 18-25
category_spending_18_25_df = purchases_18_25_with_total_cost.groupBy("category") \
    .agg(F.sum("total_cost").alias("total_spending_18_25")) \
    .orderBy(F.col("total_spending_18_25").desc())

print("\n--- Крок 4: Сума покупок за категоріями (вік 18-25) ---")
category_spending_18_25_df.show()


# --- Крок 5: Частка покупок за категоріями від сумарних витрат (вік 18-25) ---

# Обчислюємо загальну суму витрат для групи 18-25
total_spending_group_18_25 = category_spending_18_25_df.agg(
    F.sum("total_spending_18_25")
).first()[0] # щоб отримати скалярне значення

if total_spending_group_18_25 is None or total_spending_group_18_25 == 0:
     print("\nУвага: Загальні витрати для групи 18-25 дорівнюють нулю або не визначені. Неможливо розрахувати частку.")
     category_share_18_25_df = spark.createDataFrame([], category_spending_18_25_df.schema) # Створити порожній DataFrame
else:
    # Додаємо колонку з відсотком витрат
    category_share_18_25_df = category_spending_18_25_df.withColumn(
        "spending_share_percentage",
        F.round((F.col("total_spending_18_25") / total_spending_group_18_25) * 100, 2) # Округлюємо до 2 знаків
    )
    print(f"\nЗагальні витрати групи 18-25: {total_spending_group_18_25}")
    print("\n--- Крок 5: Частка витрат за категоріями (вік 18-25) ---")
    category_share_18_25_df.show()


# --- Крок 6: 3 категорії продуктів з найвищим відсотком витрат (вік 18-25) ---

# Сортуємо DataFrame з частками за спаданням відсотка та вибираємо топ-3
top_3_categories_18_25_df = category_share_18_25_df.orderBy(F.col("spending_share_percentage").desc()).limit(3)

print("\n--- Крок 6: Топ-3 категорії за часткою витрат (вік 18-25) ---")
top_3_categories_18_25_df.select("category", "spending_share_percentage").show()


# --- Завершення роботи Spark ---
print("\nАналіз завершено.")
spark.stop()