from typing import Generic, List, Literal, Optional, Type, Union, overload

from weaviate.collections.classes.filters import (
    _Filters,
)
from weaviate.collections.classes.grpc import METADATA, PROPERTIES, REFERENCES
from weaviate.collections.classes.internal import (
    _GenerativeReturn,
    _QueryReturn,
    _Generative,
    ReturnProperties,
    ReturnReferences,
    _QueryOptions,
    References,
    TReferences,
    WeaviateReferences,
)
from weaviate.collections.classes.types import Properties, TProperties
from weaviate.collections.queries.base import _BaseQuery


class _BM25Query(Generic[Properties, References], _BaseQuery[Properties, References]):
    @overload
    def bm25(
        self,
        query: str,
        query_properties: Optional[List[str]] = None,
        limit: Optional[int] = None,
        auto_limit: Optional[int] = None,
        filters: Optional[_Filters] = None,
        include_vector: bool = False,
        return_metadata: Optional[METADATA] = None,
        *,
        return_properties: Optional[PROPERTIES] = None,
        return_references: Literal[None] = None,
    ) -> _QueryReturn[Properties, References]:
        ...

    @overload
    def bm25(
        self,
        query: str,
        query_properties: Optional[List[str]] = None,
        limit: Optional[int] = None,
        auto_limit: Optional[int] = None,
        filters: Optional[_Filters] = None,
        include_vector: bool = False,
        return_metadata: Optional[METADATA] = None,
        *,
        return_properties: Optional[PROPERTIES] = None,
        return_references: REFERENCES,
    ) -> _QueryReturn[Properties, WeaviateReferences]:
        ...

    @overload
    def bm25(
        self,
        query: str,
        query_properties: Optional[List[str]] = None,
        limit: Optional[int] = None,
        auto_limit: Optional[int] = None,
        filters: Optional[_Filters] = None,
        include_vector: bool = False,
        return_metadata: Optional[METADATA] = None,
        *,
        return_properties: Optional[PROPERTIES] = None,
        return_references: Type[TReferences],
    ) -> _QueryReturn[Properties, TReferences]:
        ...

    @overload
    def bm25(
        self,
        query: str,
        query_properties: Optional[List[str]] = None,
        limit: Optional[int] = None,
        auto_limit: Optional[int] = None,
        filters: Optional[_Filters] = None,
        include_vector: bool = False,
        return_metadata: Optional[METADATA] = None,
        *,
        return_properties: Type[TProperties],
        return_references: Literal[None] = None,
    ) -> _QueryReturn[TProperties, References]:
        ...

    @overload
    def bm25(
        self,
        query: str,
        query_properties: Optional[List[str]] = None,
        limit: Optional[int] = None,
        auto_limit: Optional[int] = None,
        filters: Optional[_Filters] = None,
        include_vector: bool = False,
        return_metadata: Optional[METADATA] = None,
        *,
        return_properties: Type[TProperties],
        return_references: REFERENCES,
    ) -> _QueryReturn[TProperties, WeaviateReferences]:
        ...

    @overload
    def bm25(
        self,
        query: str,
        query_properties: Optional[List[str]] = None,
        limit: Optional[int] = None,
        auto_limit: Optional[int] = None,
        filters: Optional[_Filters] = None,
        include_vector: bool = False,
        return_metadata: Optional[METADATA] = None,
        *,
        return_properties: Type[TProperties],
        return_references: Type[TReferences],
    ) -> _QueryReturn[TProperties, TReferences]:
        ...

    def bm25(
        self,
        query: str,
        query_properties: Optional[List[str]] = None,
        limit: Optional[int] = None,
        auto_limit: Optional[int] = None,
        filters: Optional[_Filters] = None,
        include_vector: bool = False,
        return_metadata: Optional[METADATA] = None,
        *,
        return_properties: Optional[ReturnProperties[TProperties]] = None,
        return_references: Optional[ReturnReferences[TReferences]] = None,
    ) -> Union[
        _QueryReturn[Properties, References],
        _QueryReturn[Properties, WeaviateReferences],
        _QueryReturn[Properties, TReferences],
        _QueryReturn[TProperties, References],
        _QueryReturn[TProperties, WeaviateReferences],
        _QueryReturn[TProperties, TReferences],
    ]:
        """Search for objects in this collection using the keyword-based BM25 algorithm.

        See the [docs](https://weaviate.io/developers/weaviate/search/bm25) for a more detailed explanation.

        Arguments:
            `query`
                The keyword-based query to search for, REQUIRED.
            `query_properties`
                The properties to search in. If not specified, all properties are searched.
            `limit`
                The maximum number of results to return. If not specified, the default limit specified by the server is returned.
            `auto_limit`
                The maximum number of [autocut](https://weaviate.io/developers/weaviate/api/graphql/additional-operators#autocut) results to return. If not specified, no limit is applied.
            `filters`
                The filters to apply to the search.
            `include_vector`
                Whether to include the vector in the results. If not specified, this is set to False.
            `return_metadata`
                The metadata to return for each object, defaults to `None`.
            `return_properties`
                The properties to return for each object.

        NOTE:
            If `return_properties` is not provided then all non-reference properties are returned including nested properties.
            If `return_metadata` is not provided then no metadata is provided.
            If `return_references` is not provided then no references are provided.

        Returns:
            A `_QueryReturn` object that includes the searched objects.

        Raises:
            `weaviate.exceptions.WeaviateQueryException`:
                If the network connection to Weaviate fails.
        """
        res = self._query().bm25(
            query=query,
            properties=query_properties,
            limit=limit,
            autocut=auto_limit,
            filters=filters,
            return_metadata=self._parse_return_metadata(return_metadata, include_vector),
            return_properties=self._parse_return_properties(return_properties),
            return_references=self._parse_return_references(return_references),
        )
        return self._result_to_query_return(
            res,
            _QueryOptions.from_input(
                return_metadata=return_metadata,
                return_properties=return_properties,
                include_vector=include_vector,
                collection_references=self._references,
                query_references=return_references,
            ),
            return_properties,
            return_references,
        )


class _BM25Generate(Generic[Properties, References], _BaseQuery[Properties, References]):
    @overload
    def bm25(
        self,
        query: str,
        single_prompt: Optional[str] = None,
        grouped_task: Optional[str] = None,
        grouped_properties: Optional[List[str]] = None,
        query_properties: Optional[List[str]] = None,
        limit: Optional[int] = None,
        auto_limit: Optional[int] = None,
        filters: Optional[_Filters] = None,
        include_vector: bool = False,
        return_metadata: Optional[METADATA] = None,
        *,
        return_properties: Optional[PROPERTIES] = None,
        return_references: Literal[None] = None,
    ) -> _GenerativeReturn[Properties, References]:
        ...

    @overload
    def bm25(
        self,
        query: str,
        single_prompt: Optional[str] = None,
        grouped_task: Optional[str] = None,
        grouped_properties: Optional[List[str]] = None,
        query_properties: Optional[List[str]] = None,
        limit: Optional[int] = None,
        auto_limit: Optional[int] = None,
        filters: Optional[_Filters] = None,
        include_vector: bool = False,
        return_metadata: Optional[METADATA] = None,
        *,
        return_properties: Optional[PROPERTIES] = None,
        return_references: REFERENCES,
    ) -> _GenerativeReturn[Properties, WeaviateReferences]:
        ...

    @overload
    def bm25(
        self,
        query: str,
        single_prompt: Optional[str] = None,
        grouped_task: Optional[str] = None,
        grouped_properties: Optional[List[str]] = None,
        query_properties: Optional[List[str]] = None,
        limit: Optional[int] = None,
        auto_limit: Optional[int] = None,
        filters: Optional[_Filters] = None,
        include_vector: bool = False,
        return_metadata: Optional[METADATA] = None,
        *,
        return_properties: Optional[PROPERTIES] = None,
        return_references: Type[TReferences],
    ) -> _GenerativeReturn[Properties, TReferences]:
        ...

    @overload
    def bm25(
        self,
        query: str,
        single_prompt: Optional[str] = None,
        grouped_task: Optional[str] = None,
        grouped_properties: Optional[List[str]] = None,
        query_properties: Optional[List[str]] = None,
        limit: Optional[int] = None,
        auto_limit: Optional[int] = None,
        filters: Optional[_Filters] = None,
        include_vector: bool = False,
        return_metadata: Optional[METADATA] = None,
        *,
        return_properties: Type[TProperties],
        return_references: Literal[None] = None,
    ) -> _GenerativeReturn[TProperties, References]:
        ...

    @overload
    def bm25(
        self,
        query: str,
        single_prompt: Optional[str] = None,
        grouped_task: Optional[str] = None,
        grouped_properties: Optional[List[str]] = None,
        query_properties: Optional[List[str]] = None,
        limit: Optional[int] = None,
        auto_limit: Optional[int] = None,
        filters: Optional[_Filters] = None,
        include_vector: bool = False,
        return_metadata: Optional[METADATA] = None,
        *,
        return_properties: Type[TProperties],
        return_references: REFERENCES,
    ) -> _GenerativeReturn[TProperties, WeaviateReferences]:
        ...

    @overload
    def bm25(
        self,
        query: str,
        single_prompt: Optional[str] = None,
        grouped_task: Optional[str] = None,
        grouped_properties: Optional[List[str]] = None,
        query_properties: Optional[List[str]] = None,
        limit: Optional[int] = None,
        auto_limit: Optional[int] = None,
        filters: Optional[_Filters] = None,
        include_vector: bool = False,
        return_metadata: Optional[METADATA] = None,
        *,
        return_properties: Type[TProperties],
        return_references: Type[TReferences],
    ) -> _GenerativeReturn[TProperties, TReferences]:
        ...

    def bm25(
        self,
        query: str,
        single_prompt: Optional[str] = None,
        grouped_task: Optional[str] = None,
        grouped_properties: Optional[List[str]] = None,
        query_properties: Optional[List[str]] = None,
        limit: Optional[int] = None,
        auto_limit: Optional[int] = None,
        filters: Optional[_Filters] = None,
        include_vector: bool = False,
        return_metadata: Optional[METADATA] = None,
        *,
        return_properties: Optional[ReturnProperties[TProperties]] = None,
        return_references: Optional[ReturnReferences[TReferences]] = None,
    ) -> Union[
        _GenerativeReturn[Properties, References],
        _GenerativeReturn[Properties, WeaviateReferences],
        _GenerativeReturn[Properties, TReferences],
        _GenerativeReturn[TProperties, References],
        _GenerativeReturn[TProperties, WeaviateReferences],
        _GenerativeReturn[TProperties, TReferences],
    ]:
        """Perform retrieval-augmented generation (RaG) on the results of a keyword-based BM25 search of objects in this collection.

        See the [docs](https://weaviate.io/developers/weaviate/search/bm25) for a more detailed explanation.

        Arguments:
            `query`
                The keyword-based query to search for, REQUIRED.
            `single_prompt`
                The prompt to use for RaG on each object individually.
            `grouped_task`
                The prompt to use for RaG on the entire result set.
            `grouped_properties`
                The properties to use in the RaG on the entire result set.
            `query_properties`
                The properties to search in. If not specified, all properties are searched.
            `limit`
                The maximum number of results to return. If not specified, the default limit specified by the server is returned.
            `auto_limit`
                The maximum number of [autocut](https://weaviate.io/developers/weaviate/api/graphql/additional-operators#autocut) results to return. If not specified, no limit is applied.
            `filters`
                The filters to apply to the search.
            `include_vector`
                Whether to include the vector in the results. If not specified, this is set to False.
            `return_metadata`
                The metadata to return for each object, defaults to `None`.
            `return_properties`
                The properties to return for each object.
            `return_references`
                The references to return for each object.

        NOTE:
            If `return_properties` is not provided then all non-reference properties are returned including nested properties.
            If `return_metadata` is not provided then no metadata is provided.
            If `return_references` is not provided then no references are provided.

        Returns:
            A `_GenerativeReturn` object that includes the searched objects with per-object generated results and group generated results.

        Raises:
            `weaviate.exceptions.WeaviateQueryException`:
                If the network connection to Weaviate fails.
        """
        res = self._query().bm25(
            query=query,
            properties=query_properties,
            limit=limit,
            autocut=auto_limit,
            filters=filters,
            return_metadata=self._parse_return_metadata(return_metadata, include_vector),
            return_properties=self._parse_return_properties(return_properties),
            return_references=self._parse_return_references(return_references),
            generative=_Generative(
                single=single_prompt,
                grouped=grouped_task,
                grouped_properties=grouped_properties,
            ),
        )
        return self._result_to_generative_return(
            res,
            _QueryOptions.from_input(
                return_metadata=return_metadata,
                return_properties=return_properties,
                include_vector=include_vector,
                collection_references=self._references,
                query_references=return_references,
            ),
            return_properties,
            return_references,
        )
