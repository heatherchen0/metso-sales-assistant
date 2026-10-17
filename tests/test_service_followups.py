"""Service follow-ups use typed hours and preserve checked evidence in concise answers."""
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest

from metso_assistant.request_scope import annotate_material_supply
from metso_assistant.service_answer import service_answer
from metso_assistant.service_evidence import ServiceWorkspace
from metso_assistant.service_request import resolve_duration_inputs
from metso_assistant.service_travel import travel_context
from metso_assistant.services import parse_text
from metso_assistant.store import build_store


class ServiceFollowupTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.serial=0

    def job(self,travel_text=None,owned=False):
        self.serial+=1
        record=parse_text("""Reference: FSR-9951
Site: Invented component facility, Finland
Equipment: Synthetic separator
On-site hours: 3
Travel hours: 1
| Task | Time Spent (hrs) |
|------|------------------|
| Replace quartz injectors | 2 |

| Item | Quantity | Unit Rate (EUR) | Total (EUR) |
|------|----------|-----------------|-------------|
| Quartz injectors | 2 | 10.25/ea | 20.50 |
| Labour | 3 hrs | 50/hr | 150 |
| Travel | 1 hr | 15/hr | 15 |
""",source_path="invented/FSR-9951.txt")
        messages=["Replace 5 quartz injectors."]
        if owned:messages.append("The customer already has the quartz injectors.")
        if travel_text:messages.append(travel_text)
        inputs=[{"id":"U001","kind":"user_input","value":5,"unit":"each","user_text":messages[0],
                 "scope":{"role":"requested_quantity","turn":0,"entity_text":"quartz injectors"}}]
        inputs=annotate_material_supply(inputs,messages)
        durations=resolve_duration_inputs(messages)
        inputs.extend(fact for index,item in enumerate(durations,2) if (fact:=item.to_fact("U"+str(index))) is not None)
        store=self.root/f"job-{self.serial}.sqlite";build_store(store,[],[],[],[record])
        workspace=ServiceWorkspace(store,self.root/f"index-{self.serial}.sqlite",inputs=inputs)
        workspace.read_report("FSR-9951")
        return workspace,messages,durations

    def material(self,workspace):
        rate=next(f for f in workspace.facts.values() if f["kind"]=="unit_price")
        return workspace.calculate("multiply",[rate["id"],"U001"])

    def work(self,workspace):
        facts=list(workspace.facts.values())
        task=next(f for f in facts if f["kind"]=="task_duration")
        count=next(f for f in facts if f["kind"]=="quantity" and f["scope"].get("description")=="Quartz injectors")
        duration=workspace.calculate("scale_duration",[task["id"],"U001",count["id"]])["facts"][0]
        rate=next(f for f in facts if f["kind"]=="hourly_rate" and f["scope"].get("category")=="labour")
        return workspace.calculate("multiply",[rate["id"],duration["id"]])

    def ranged_job(self, text="Round-trip travel will be 4 hours.", *, base_currency="USD", equal_rates=False):
        self.serial += 1
        records=[]
        entries=[("FSR-9931","Chile",base_currency,7 if base_currency=="EUR" else 10,"Synthetic separator"),
                 ("FSR-9932","Chile","USD",10 if equal_rates else 20,"Unrelated test conveyor"),
                 ("FSR-9933","Chile","USD",10 if equal_rates else 15,"Synthetic pump"),
                 ("FSR-9934","Finland","USD",99,"Foreign test separator")]
        for index,(report,country,currency,rate,equipment) in enumerate(entries):
            tasks=("On-site hours: 3\n| Task | Time Spent (hrs) |\n|---|---|\n| Replace quartz injectors | 2 |\n\n" if index==0 else "")
            component="| Quartz injectors | 2 | 3/ea | 6 |\n| Labour | 3 hrs | 50/hr | 150 |\n" if index==0 else ""
            records.append(parse_text(f"Reference: {report}\nSite: Invented facility, {country}\nEquipment: {equipment}\n{tasks}| Item | Quantity | Unit Rate ({currency}) | Total ({currency}) |\n|---|---|---|---|\n{component}| Travel | 1 hr | {rate}/hr | {rate} |\n",source_path=f"invented/{report}.txt"))
        messages=["A customer in Chile needs 5 quartz injectors replaced.",text]
        durations=resolve_duration_inputs(messages)
        inputs=[{"id":"U001","kind":"user_input","value":5,"unit":"each","user_text":messages[0],
                 "scope":{"role":"requested_quantity","turn":0,"entity_text":"quartz injectors"}}]
        inputs.extend(fact for index,record in enumerate(durations,2) if (fact:=record.to_fact("U"+str(index))) is not None)
        store=self.root/f"range-{self.serial}.sqlite";build_store(store,[],[],[],records)
        workspace=ServiceWorkspace(store,self.root/f"range-index-{self.serial}.sqlite",inputs=inputs)
        workspace.read_report("FSR-9931")
        duration=next(record for record in durations if record.role=="travel_hours")
        context,_=travel_context(workspace,"Chile",duration)
        return workspace,messages,durations,context

    def render_range(self,workspace,messages,durations,context,categories=("material","labour","travel")):
        return service_answer(workspace,[],messages,known={"country":"Chile"},
            plan={"purpose":"requested_work","requested_categories":list(categories)},
            duration_inputs=durations,travel_context=context)

    def test_country_travel_interval_is_two_scenarios_not_two_charges(self):
        workspace,messages,durations,context=self.ranged_job()
        self.material(workspace); self.work(workspace)
        result,operations=self.render_range(workspace,messages,durations,context)
        data=result["data"]; overall=data["subtotal_ranges_by_currency"]["USD"]
        service=data["category_subtotal_ranges_by_currency"]["service"]["USD"]
        self.assertEqual((overall["minimum"],overall["maximum"]),(Decimal(305),Decimal(345)))
        self.assertEqual((service["minimum"],service["maximum"]),(Decimal(290),Decimal(330)))
        self.assertNotIn("USD",data["subtotal_refs_by_currency"])
        self.assertEqual(workspace.facts[data["category_subtotal_refs_by_currency"]["material"]["USD"]]["value"],Decimal(15))
        endpoints=set(context["endpoint_fact_ids"])
        self.assertFalse(any(endpoints <= set(fact.get("parents",[])) for fact in workspace.facts.values() if fact.get("scope",{}).get("operation")=="sum"))
        self.assertEqual(result["status"],"ok");self.assertFalse(data["unpriced_categories"])
        self.assertIn("Historical travel rate (Chile): **USD 10.00–20.00/h",data["summary"])
        self.assertIn("billed travel for 4 h",data["summary"])
        self.assertNotIn("mixed equipment",data["summary"])
        self.assertNotIn("Travel uses the observed range",data["summary"])
        self.assertIn("Travel uses the observed range across these reports.",data["disclosures"])
        self.assertEqual({source["report_id"] for source in result["sources"]},{"FSR-9931","FSR-9932","FSR-9933"})
        self.assertTrue(all(reference in data["selected_fact_ids"] for reference in context["cohorts"][0]["fact_ids"]))
        self.assertTrue(all(not entry["result"].get("error") for entry in operations))

    def test_point_country_travel_preserves_full_legacy_scalar_total(self):
        workspace,messages,durations,context=self.ranged_job(equal_rates=True)
        self.material(workspace);self.work(workspace)
        result,_=self.render_range(workspace,messages,durations,context)
        data=result["data"]
        self.assertEqual(workspace.facts[data["subtotal_refs_by_currency"]["USD"]]["value"],Decimal(305))
        self.assertEqual(workspace.facts[data["category_subtotal_refs_by_currency"]["service"]["USD"]]["value"],Decimal(290))
        self.assertEqual(data["subtotal_ranges_by_currency"]["USD"]["minimum"],data["subtotal_ranges_by_currency"]["USD"]["maximum"])
        self.assertEqual(len(context["endpoint_fact_ids"]),1)

    def test_mixed_currency_with_materials_line_retains_every_priced_currency(self):
        workspace,messages,durations,context=self.ranged_job(base_currency="EUR")
        self.material(workspace);self.work(workspace)
        result,_=self.render_range(workspace,messages,durations,context)
        line=next(paragraph for paragraph in result["data"]["summary"].split("\n\n") if paragraph.startswith("With these materials:"))
        self.assertIn("EUR 293.00",line);self.assertIn("USD 60.00–80.00",line)
        ranges=result["data"]["subtotal_ranges_by_currency"]
        self.assertEqual(set(ranges),{"EUR","USD"})
        self.assertEqual((ranges["EUR"]["minimum"],ranges["EUR"]["maximum"]),(Decimal(293),Decimal(293)))
        self.assertIn("exchange rate",result["data"]["summary"])

    def test_unresolved_hours_keep_country_rate_interval_without_monetary_bounds(self):
        workspace,messages,durations,context=self.ranged_job("Travel time will be 8 hours.")
        self.material(workspace);self.work(workspace)
        result,_=self.render_range(workspace,messages,durations,context)
        self.assertEqual(workspace.facts[result["data"]["subtotal_refs_by_currency"]["USD"]]["value"],Decimal(265))
        self.assertEqual(result["data"]["subtotal_ranges_by_currency"]["USD"]["minimum"],Decimal(265))
        self.assertEqual(result["data"]["subtotal_ranges_by_currency"]["USD"]["maximum"],Decimal(265))
        self.assertIn("USD 10.00–20.00/h",result["data"]["summary"])
        self.assertNotIn("billed travel for",result["data"]["summary"])
        self.assertIn("8 hours",result["question"])
        self.assertEqual(result["missing_fields"],["round_trip_travel_hours"])
        self.assertEqual(result["data"]["unpriced_categories"],["travel"])

    def test_missing_country_or_local_cohort_never_uses_loaded_foreign_travel_rates(self):
        for context,expected in (({"status":"missing_country","country":None,"cohorts":[],"endpoint_fact_ids":[]},"require the service country"),
                                 ({"status":"unavailable","country":"Canada","cohorts":[],"endpoint_fact_ids":[]},"No validated historical travel rate is available for Canada")):
            workspace,messages,durations,_=self.ranged_job("Travel hours are unknown.")
            result,_=self.render_range(workspace,messages,durations,context,categories=("travel",))
            self.assertIn(expected,result["data"]["summary"])
            self.assertNotIn("/h",result["data"]["summary"])
            self.assertFalse(result["data"]["historical_hourly_rate_cohorts"])
            self.assertFalse(result["data"]["subtotal_ranges_by_currency"])
            self.assertFalse(result["sources"])

    def test_material_only_excludes_even_precomputed_travel_endpoints(self):
        workspace,messages,durations,context=self.ranged_job()
        self.material(workspace);self.work(workspace)
        result,_=self.render_range(workspace,messages,durations,context,categories=("material",))
        self.assertEqual(workspace.facts[result["data"]["subtotal_refs_by_currency"]["USD"]]["value"],Decimal(15))
        self.assertEqual(result["data"]["subtotal_ranges_by_currency"]["USD"]["minimum"],Decimal(15))
        self.assertEqual(result["data"]["subtotal_ranges_by_currency"]["USD"]["maximum"],Decimal(15))
        self.assertFalse(result["data"]["historical_hourly_rate_cohorts"])
        self.assertNotIn("Historical travel rate",result["data"]["summary"])
        self.assertFalse(result["data"]["unpriced_categories"])

    def test_ambiguous_eight_keeps_partial_amount_and_shows_rate_question_and_pending_field(self):
        workspace,messages,durations=self.job("The travel time will be 8 hours.")
        self.material(workspace)
        result,_=service_answer(workspace,[],messages,known={"country":"Chile"},plan={"purpose":"requested_work","requested_categories":["material","labour","travel"]},duration_inputs=durations)
        summary=result["data"]["summary"]
        self.assertEqual(result["status"],"ok");self.assertIn("51.25",summary)
        self.assertIn("EUR 15.00/h",summary);self.assertIn("8 hours",result["question"])
        self.assertEqual(result["missing_fields"],["round_trip_travel_hours"])
        self.assertIn(result["question"],summary);self.assertIn("Foreign-country",summary)
        self.assertEqual(result["data"]["unpriced_categories"],["labour","travel"])
        self.assertLess(len(summary.split()),230)

    def test_known_travel_recalculates_work_scope_with_owned_materials_and_retains_task_quality(self):
        totals=[]
        for hours in (4,8):
            workspace,messages,durations=self.job(f"Total travel will be {hours} hours.",owned=True)
            self.assertIn("error",self.material(workspace));self.work(workspace)
            travel=next(f for f in workspace.facts.values() if f["kind"]=="user_input" and f["scope"].get("role")=="travel_hours")
            rate=next(f for f in workspace.facts.values() if f["kind"]=="hourly_rate" and f["scope"].get("category")=="travel")
            workspace.calculate("multiply",[rate["id"],travel["id"]])
            result,_=service_answer(workspace,[],messages,known={"country":"Chile"},plan={"purpose":"requested_work","requested_categories":["labour","travel"]},duration_inputs=durations)
            subtotal=workspace.facts[result["data"]["subtotal_refs_by_currency"]["EUR"]]
            totals.append(subtotal["value"])
            self.assertIn("Labour (estimated hours)",result["data"]["summary"])
            self.assertIn("Estimated hours cover one historical task; source task totals do not reconcile.",result["data"]["summary"])
            self.assertNotIn("Estimated labour hours scale",result["data"]["summary"])
            self.assertTrue(any("Estimated labour hours scale a recorded task" in note for note in result["data"]["disclosures"]))
            self.assertEqual(result["data"]["labour_estimation_basis"][0]["task_label"],"Replace quartz injectors")
            self.assertIn("Customer-supplied",result["data"]["summary"])
            self.assertIn("do not reconcile",result["data"]["summary"])
            self.assertTrue(result["data"]["assumptions"])
            self.assertFalse(result["data"]["unpriced_categories"])
        self.assertEqual(totals,[Decimal("310.00"),Decimal("370.00")])

    def test_visit_base_precedes_conditional_material_addition_and_combined_checked_subtotal(self):
        workspace,messages,durations=self.job("Total billed travel will be 4 hours.")
        self.material(workspace);self.work(workspace)
        travel=next(f for f in workspace.facts.values() if f["kind"]=="user_input" and f["scope"].get("role")=="travel_hours")
        rate=next(f for f in workspace.facts.values() if f["kind"]=="hourly_rate" and f["scope"].get("category")=="travel")
        workspace.calculate("multiply",[rate["id"],travel["id"]])
        result,_=service_answer(workspace,[],messages,known={"country":"Chile"},plan={"purpose":"requested_work","requested_categories":["material","labour","travel"]},duration_inputs=durations)
        groups=result["data"]["category_subtotal_refs_by_currency"]
        self.assertEqual(workspace.facts[groups["service"]["EUR"]]["value"],Decimal("310.00"))
        self.assertEqual(workspace.facts[groups["material"]["EUR"]]["value"],Decimal("51.25"))
        self.assertEqual(workspace.facts[result["data"]["subtotal_refs_by_currency"]["EUR"]]["value"],Decimal("361.25"))
        summary=result["data"]["summary"]
        self.assertLess(summary.index("310.00"),summary.index("Optional material addition"))
        self.assertLess(summary.index("Optional material addition"),summary.index("With these materials"))
        self.assertIn("if these materials must be purchased",summary)
        self.assertLess(len(summary.split()),230)

    def test_owned_equipment_anchor_does_not_claim_requested_materials_are_customer_supplied(self):
        workspace,messages,durations=self.job()
        text="We have 2 separators; replace 5 quartz injectors for each separator."
        workspace.facts["U001"]["user_text"]=text
        workspace.facts["U_EQUIPMENT"]={"id":"U_EQUIPMENT","kind":"user_input","value":Decimal(2),"unit":"each","user_text":text,
                                       "scope":{"role":"requested_quantity","turn":0,"entity_text":"separators","customer_supplied":True}}
        result,_=service_answer(workspace,[],[text],plan={"purpose":"requested_work","requested_categories":["material"],"equipment_count_ref":"U_EQUIPMENT"})
        self.assertNotIn("Customer-supplied materials",result["data"]["summary"])
        # Proven repetition anchors are excluded even if the plan omits its optional anchor.
        result,_=service_answer(workspace,[],[text],plan={"purpose":"requested_work","requested_categories":["material"]})
        self.assertNotIn("Customer-supplied materials",result["data"]["summary"])

    def test_material_only_subtotal_acknowledges_supplied_or_unknown_duration_context_without_pricing_it(self):
        for text,expected in (("Total travel will be 4 hours.","Travel: 4 h total supplied"),
                              ("Labour hours are unknown; travel hours are unknown.","Labour hours remain unknown; Travel hours remain unknown")):
            with self.subTest(text=text):
                workspace,messages,durations=self.job(text)
                self.material(workspace)
                result,_=service_answer(workspace,[],messages,plan={"purpose":"requested_work","requested_categories":["material"]},duration_inputs=durations)
                subtotal=workspace.facts[result["data"]["subtotal_refs_by_currency"]["EUR"]]
                self.assertEqual(subtotal["value"],Decimal("51.25"))
                self.assertIn("Outside this subtotal",result["data"]["summary"])
                self.assertIn(expected,result["data"]["summary"])
                self.assertFalse(result["data"]["historical_hourly_rate_cohorts"])
                self.assertFalse(result["data"]["unpriced_categories"])
                self.assertIsNone(result["question"])

    def test_original_task_example_keeps_its_task_label_and_material_only_scope_suppresses_unrequested_rates(self):
        workspace,messages,durations=self.job()
        task=next(f for f in workspace.facts.values() if f["kind"]=="task_duration")
        rate=next(f for f in workspace.facts.values() if f["kind"]=="hourly_rate" and f["scope"].get("category")=="labour")
        workspace.calculate("multiply",[rate["id"],task["id"]])
        result,_=service_answer(workspace,[],messages,plan={"purpose":"requested_work","requested_categories":["labour"]})
        self.assertIn("original task example",result["data"]["summary"])
        self.assertIn("labour task subtotal",result["data"]["summary"])
        self.assertNotIn("labour and billed travel subtotal",result["data"]["summary"])
        partial,_=service_answer(workspace,[],messages,plan={"purpose":"requested_work","requested_categories":["labour","travel"]})
        self.assertIn("labour task subtotal; travel not yet priced",partial["data"]["summary"])
        other,messages,durations=self.job("Travel will be 8 hours.")
        self.material(other)
        result,_=service_answer(other,[],messages,plan={"purpose":"requested_work","requested_categories":["material"]},duration_inputs=durations)
        self.assertIsNone(result["question"])
        self.assertFalse(result["data"]["historical_hourly_rate_cohorts"])

    def test_hourly_ranges_keep_currency_and_location_cohorts_separate_and_do_not_create_costs(self):
        records=[]
        for index,(country,currency,rate) in enumerate((("Chile","USD",10),("Chile","USD",20),("Chile","EUR",7),("Finland","USD",99)),1):
            records.append(parse_text(f"Reference: FSR-994{index}\nSite: Invented site, {country}\nEquipment: Synthetic separator\n| Item | Quantity | Unit Rate ({currency}) | Total ({currency}) |\n|---|---|---|---|\n| Travel | 1 hr | {rate}/hr | {rate} |\n",source_path=f"invented/FSR-994{index}.txt"))
        store=self.root/"rates.sqlite";build_store(store,[],[],[],records)
        workspace=ServiceWorkspace(store,self.root/"rates-index.sqlite")
        for record in records:workspace.read_report(record["report_id"])
        result,_=service_answer(workspace,[],["Travel time is unknown."],known={"country":"Chile"},plan={"purpose":"requested_work","requested_categories":["travel"]})
        cohorts={(group["country"],group["currency"]):(group["minimum"],group["maximum"]) for group in result["data"]["historical_hourly_rate_cohorts"]}
        self.assertEqual(cohorts,{("Chile","USD"):(Decimal(10),Decimal(20)),("Chile","EUR"):(Decimal(7),Decimal(7)),("Finland","USD"):(Decimal(99),Decimal(99))})
        self.assertIn("USD 10.00–20.00/h",result["data"]["summary"])
        self.assertIn("exchange rate",result["data"]["summary"])
        self.assertFalse(result["data"]["subtotal_refs_by_currency"])
        self.assertFalse(any(f["kind"]=="calculation" for f in workspace.facts.values()))
        self.assertIsNone(result["question"])


if __name__ == "__main__":unittest.main()
