library(lubridate)
library(glue)

# Loading calendar
tz <- "UTC"
ts <- 60*60*24
FR_calendar <- generate_calendar(
  date_begin = "2018-01-01 00:00",
  "2021-12-31 23:50",
  ts = ts,
  tz = tz,
  holidays = "FR"
)

moment <- seq.Date(date("2018-01-01"), date("2021-12-31"), "days")

for (zone in c('A', 'B', 'C')) {
  FR_calendar[glue("holiday_{zone}")] <- as.integer(is_school_break(moment, zone=zone))
}

FR_calendar <- FR_calendar %>% 
  select(
    -isoSemaine,
    -isoAnnee,
    -Mois,
    -Jour,
    -Heure,
    -Minute,
    -Instant,
    -Posan,
    -Tendance,
    )

write.csv(FR_calendar, "data-raw/fr_jour_feries.csv")
